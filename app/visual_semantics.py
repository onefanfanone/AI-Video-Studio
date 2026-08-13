from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .asset_sources import PROVIDER_TRUST
from .deepseek_planner import DeepSeekPlannerError, request_json_object


class SemanticReviewError(RuntimeError):
    pass


SEMANTIC_METADATA_KEYS = {
    "culture",
    "period",
    "dynasty",
    "reign",
    "objectdate",
    "objectbegindate",
    "objectenddate",
    "objectname",
    "medium",
    "classification",
    "description",
    "imagedescription",
    "categories",
    "tags",
    "date",
    "place",
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        value = value["value"]
    if isinstance(value, str):
        return value[:400]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        result = []
        for item in value[:8]:
            if isinstance(item, dict):
                compact = {
                    str(key): _safe_value(child)
                    for key, child in list(item.items())[:4]
                    if not str(key).lower().endswith("url")
                }
                result.append(compact)
            else:
                result.append(_safe_value(item))
        return result
    return str(value)[:400]


def compact_semantic_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if lowered in SEMANTIC_METADATA_KEYS and lowered not in found:
                    found[lowered] = _safe_value(value)
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node[:20]:
                walk(item)

    walk(candidate.get("raw_metadata") or {})
    return {
        "asset_id": str(candidate.get("asset_id") or ""),
        "provider": str(candidate.get("provider") or ""),
        "title": str(candidate.get("title") or "")[:300],
        "creator": str(candidate.get("creator") or "")[:200],
        "institution": str(candidate.get("institution") or "")[:200],
        "metadata": found,
    }


def _technical_components(candidate: dict[str, Any]) -> dict[str, int]:
    provider_trust = PROVIDER_TRUST.get(str(candidate.get("provider")), 0)
    institution = round(min(10, max(0, provider_trust / 24 * 10)))
    width = int(candidate.get("width") or 0)
    height = int(candidate.get("height") or 0)
    if max(width, height) >= 1600 and min(width, height) >= 800:
        resolution = 10
    elif width and height:
        resolution = 6
    else:
        resolution = 0
    ratio = width / height if height else 1.0
    crop = max(0, round(10 - abs(math.log(max(ratio, 0.05) / (9 / 16))) * 3.5))
    compact = compact_semantic_metadata(candidate)
    metadata_points = 0
    if compact["title"] and compact["title"].lower() != "untitled":
        metadata_points += 3
    if compact["creator"] and compact["creator"].lower() != "unknown":
        metadata_points += 2
    metadata_points += min(5, len(compact["metadata"]))
    return {
        "institution_trust": institution,
        "resolution": resolution,
        "vertical_crop": crop,
        "metadata_completeness": min(10, metadata_points),
    }


def _validate_judgments(
    response: Any, expected: dict[tuple[int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or set(response) != {"judgments"}:
        raise ValueError("顶层必须只包含 judgments。")
    rows = response.get("judgments")
    if not isinstance(rows, list):
        raise ValueError("judgments 必须为数组。")
    validated: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    required = {
        "shot_id",
        "asset_id",
        "relevance_score",
        "period_score",
        "subject_score",
        "visual_fit_score",
        "verdict",
        "conflicts",
        "reason",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("候选 judgment 字段不符合约定。")
        key = (int(row.get("shot_id", -1)), str(row.get("asset_id") or ""))
        if key not in expected or key in seen:
            raise ValueError(f"judgment 引用了未知或重复候选：{key}")
        for field in (
            "relevance_score",
            "period_score",
            "subject_score",
            "visual_fit_score",
        ):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
                raise ValueError(f"{key} 的 {field} 必须是 0 到 100 的整数。")
        if row.get("verdict") not in {"eligible", "reject"}:
            raise ValueError(f"{key} 的 verdict 必须是 eligible 或 reject。")
        if not isinstance(row.get("conflicts"), list) or any(
            not isinstance(item, str) for item in row["conflicts"]
        ):
            raise ValueError(f"{key} 的 conflicts 必须为字符串数组。")
        if not isinstance(row.get("reason"), str):
            raise ValueError(f"{key} 的 reason 必须为字符串。")
        seen.add(key)
        validated.append(dict(row))
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"DeepSeek 漏评了 {len(missing)} 个候选。")
    return validated


def _apply_judgment(
    candidate: dict[str, Any], judgment: dict[str, Any], reject_below: int
) -> None:
    semantic_score = int(judgment["relevance_score"])
    components = _technical_components(candidate)
    final_score = round(
        semantic_score * 0.60
        + components["institution_trust"]
        + components["resolution"]
        + components["vertical_crop"]
        + components["metadata_completeness"]
    )
    rejected = (
        semantic_score < reject_below
        or judgment["verdict"] == "reject"
        or bool(judgment["conflicts"])
    )
    candidate["semantic_status"] = "rejected" if rejected else "reviewed"
    candidate["semantic_requires_override"] = rejected
    candidate["semantic_review"] = dict(judgment)
    candidate["semantic_score"] = semantic_score
    candidate["score"] = min(100, max(0, final_score))
    candidate["score_detail"] = {
        "semantic_relevance": round(semantic_score * 0.60),
        **components,
    }


def _mark_unavailable(candidate: dict[str, Any], message: str) -> None:
    candidate["semantic_status"] = "unavailable"
    candidate["semantic_requires_override"] = False
    candidate["semantic_score"] = None
    candidate["semantic_review"] = {
        "verdict": "unavailable",
        "conflicts": [],
        "reason": message,
    }
    components = _technical_components(candidate)
    candidate["score"] = sum(components.values())
    candidate["score_detail"] = components


def review_asset_candidates(
    scene_plan: dict[str, Any],
    candidates: dict[str, Any],
    planner_config: dict[str, Any],
    search_config: dict[str, Any],
    env: dict[str, str],
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    review_config = dict(search_config.get("semantic_review", {}))
    if not review_config.get("enabled", True):
        candidates["semantic_review_status"] = "disabled"
        return candidates, {"status": "disabled", "batches": []}, []
    provider = str(review_config.get("provider", "deepseek")).lower()
    if provider not in {"deepseek", "chat_completions"}:
        raise SemanticReviewError(
            f"当前只支持 DeepSeek 候选语义复核，收到：{provider}"
        )
    shots_per_batch = int(review_config.get("shots_per_batch", 4))
    if not 1 <= shots_per_batch <= 8:
        raise SemanticReviewError("shots_per_batch 必须在 1 到 8 之间。")
    reject_below = int(review_config.get("reject_below", 40))
    recommendation_threshold = int(
        review_config.get(
            "recommendation_threshold",
            search_config.get("recommendation_threshold", 75),
        )
    )
    failure_policy = str(review_config.get("failure_policy", "manual_review")).lower()
    settings = {**planner_config, **review_config}
    intents = {int(item["shot_id"]): item for item in scene_plan.get("shots", [])}
    for shot in candidates.get("shots", []):
        intent = intents.get(int(shot["shot_id"]), {})
        shot["time_context"] = intent.get("time_context", {})
        shot["must_include"] = list(intent.get("must_include", []))
        shot["avoid"] = list(intent.get("avoid", []))
        shot["semantic_review_status"] = "pending"
        for candidate in shot.get("candidates", []):
            candidate["semantic_metadata"] = compact_semantic_metadata(candidate)
            if not candidate.get("selectable") or candidate.get("provider") == "openverse":
                candidate["semantic_status"] = "hard_rejected"
                candidate["semantic_requires_override"] = False

    shot_rows = list(candidates.get("shots", []))
    batch_reports: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    failure_count = 0
    system_prompt = (
        "你是博物馆素材的文字元数据相关性审校员。只根据给定镜头意图和候选元数据判断，"
        "不得补充外部史实，也不能把机构可信度或高分辨率当成语义相关。逐项判断时代/文化、"
        "主题物件、画面可见性和竖屏短视频适配。只输出 JSON，顶层只能有 judgments；每个"
        "候选必须且只能返回一次。verdict 只能是 eligible 或 reject。候选拍摄/数字化年份本身"
        "不是拒绝理由；应判断它实际描绘或记录的对象。"
    )
    for batch_start in range(0, len(shot_rows), shots_per_batch):
        batch_shots = shot_rows[batch_start : batch_start + shots_per_batch]
        expected: dict[tuple[int, str], dict[str, Any]] = {}
        payload_shots: list[dict[str, Any]] = []
        for shot in batch_shots:
            shot_id = int(shot["shot_id"])
            intent = intents[shot_id]
            compact_candidates = []
            for candidate in shot.get("candidates", []):
                if candidate.get("semantic_status") == "hard_rejected" or candidate.get(
                    "ai_generated"
                ):
                    continue
                compact = compact_semantic_metadata(candidate)
                compact_candidates.append(compact)
                expected[(shot_id, str(candidate["asset_id"]))] = candidate
            payload_shots.append(
                {
                    "shot_id": shot_id,
                    "narration": intent.get("narration", ""),
                    "time_context": intent.get("time_context", {}),
                    "location": intent.get("location", ""),
                    "people": intent.get("people", []),
                    "objects": intent.get("objects", []),
                    "action": intent.get("action", ""),
                    "must_include": intent.get("must_include", []),
                    "avoid": intent.get("avoid", []),
                    "candidates": compact_candidates,
                }
            )
        if not expected:
            for shot in batch_shots:
                shot["semantic_review_status"] = "reviewed"
            continue
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "prompt_version": review_config.get(
                        "prompt_version", "asset-semantic-v1"
                    ),
                    "model": settings.get("model", "deepseek-v4-flash"),
                    "shots": payload_shots,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = cache_root / "visuals" / "semantic-review" / f"{cache_key}.json"
        rows: list[dict[str, Any]] | None = None
        safe: dict[str, Any] = {}
        error = ""
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                rows = _validate_judgments(cached, expected)
                safe = {"cache_hit": True, "cache_key": cache_key}
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                error = str(exc)
                rows = None
        if rows is None:
            for attempt in range(3):
                try:
                    response, safe = request_json_object(
                        system_prompt,
                        {
                            "instruction": "Review every candidate and return JSON only.",
                            "shots": payload_shots,
                            "previous_validation_error": error if attempt else "",
                        },
                        settings,
                        env.get(
                            str(
                                review_config.get("secret_ref")
                                or planner_config.get("secret_ref")
                                or "DEEPSEEK_API_KEY"
                            ),
                            "",
                        ),
                    )
                    rows = _validate_judgments(response, expected)
                    _atomic_json(cache_path, response)
                    artifacts.append(cache_path)
                    break
                except (DeepSeekPlannerError, ValueError) as exc:
                    error = str(exc)
                    rows = None
            if rows is None and failure_policy != "manual_review":
                raise SemanticReviewError(
                    f"DeepSeek 候选语义复核失败：{error}"
                )
        if rows is None:
            failure_count += 1
            for candidate in expected.values():
                _mark_unavailable(candidate, error or "DeepSeek 语义复核不可用。")
            for shot in batch_shots:
                shot["semantic_review_status"] = "unavailable"
                shot["recommended_asset_id"] = None
            batch_reports.append(
                {
                    "batch": batch_start // shots_per_batch + 1,
                    "status": "unavailable",
                    "message": error,
                }
            )
            continue
        for row in rows:
            _apply_judgment(
                expected[(int(row["shot_id"]), str(row["asset_id"]))],
                row,
                reject_below,
            )
        for shot in batch_shots:
            shot["semantic_review_status"] = "reviewed"
        batch_reports.append(
            {
                "batch": batch_start // shots_per_batch + 1,
                "status": "reviewed",
                "candidate_count": len(rows),
                "response": safe,
            }
        )

    previous: str | None = None
    previous_hash: str | None = None
    force_manual = scene_plan.get("semantic_audit_status") == "unavailable"
    for shot in shot_rows:
        eligible = sorted(
            (
                item
                for item in shot.get("candidates", [])
                if item.get("selectable")
                and item.get("semantic_status") == "reviewed"
                and int(item.get("score", 0)) >= recommendation_threshold
                and item.get("asset_id") != previous
                and (
                    not previous_hash
                    or not item.get("perceptual_hash")
                    or item.get("perceptual_hash") != previous_hash
                )
            ),
            key=lambda item: (-int(item.get("score", 0)), str(item.get("asset_id"))),
        )
        recommendation = None if force_manual else (
            eligible[0]["asset_id"] if eligible else None
        )
        if shot.get("semantic_review_status") == "unavailable":
            recommendation = None
        shot["recommended_asset_id"] = recommendation
        previous = recommendation
        selected = next(
            (
                item
                for item in shot.get("candidates", [])
                if item.get("asset_id") == recommendation
            ),
            None,
        )
        previous_hash = selected.get("perceptual_hash") if selected else None
    status = "reviewed"
    if failure_count == len(range(0, len(shot_rows), shots_per_batch)):
        status = "unavailable"
    elif failure_count:
        status = "partial"
    if force_manual and status == "reviewed":
        status = "plan_unavailable"
    candidates["semantic_review_status"] = status
    candidates["plan_semantic_status"] = scene_plan.get(
        "semantic_audit_status", "unknown"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "provider": provider,
        "model": settings.get("model", "deepseek-v4-flash"),
        "reject_below": reject_below,
        "recommendation_threshold": recommendation_threshold,
        "batches": batch_reports,
    }
    return candidates, report, artifacts
