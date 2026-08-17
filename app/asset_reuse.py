from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import threading
import urllib.parse
import webbrowser
from datetime import datetime
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .asset_sources import AssetSourceError, verify_image_bytes
from .deepseek_planner import DeepSeekPlannerError, request_json_object


class AssetReuseError(RuntimeError):
    """A safe, user-facing error from the derived-project reuse layer."""


PROMPT_VERSION = "asset-reuse-match-v1"
REQUIRED_PARENT_FILES = (
    "task.json",
    "scene_plan.json",
    "asset_candidates.json",
    "asset_selection.json",
    "assets_manifest.json",
    "license_audit.json",
    "licenses.csv",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetReuseError(f"无法读取父任务产物：{path.name}。") from exc
    if not isinstance(payload, dict):
        raise AssetReuseError(f"父任务产物不是 JSON 对象：{path.name}。")
    return payload


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AssetReuseError("父任务路径不在当前工作区输出目录中。") from exc
    return candidate


def resolve_parent_task(outputs_root: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(task_id or "")):
        raise AssetReuseError("父任务 ID 无效。")
    run_dir = _inside(outputs_root, outputs_root / task_id)
    task_path = run_dir / "task.json"
    if not task_path.is_file():
        raise AssetReuseError("找不到所选父任务。")
    task = _json(task_path)
    if task.get("status") != "succeeded":
        raise AssetReuseError("只有成功完成的任务才能派生复用版本。")
    options = task.get("options") if isinstance(task.get("options"), dict) else {}
    if options.get("visual_mode") != "sourced":
        raise AssetReuseError("只有 sourced 任务才能建立画面复用池。")
    missing = [name for name in REQUIRED_PARENT_FILES if not (run_dir / name).is_file()]
    if missing:
        raise AssetReuseError("父任务台账不完整，缺少：" + "、".join(missing))
    return run_dir


def list_reusable_tasks(outputs_root: Path, project_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not outputs_root.is_dir():
        return result
    for task_path in outputs_root.glob("*/task.json"):
        try:
            data = _json(task_path)
            options = data.get("options") if isinstance(data.get("options"), dict) else {}
            project_id = str(data.get("project_id") or "")
            if (
                data.get("status") != "succeeded"
                or options.get("visual_mode") != "sourced"
                or not (project_root / project_id / "script.txt").is_file()
                or any(not (task_path.parent / name).is_file() for name in REQUIRED_PARENT_FILES)
            ):
                continue
            manifest = _json(task_path.parent / "assets_manifest.json")
            assets = manifest.get("assets") if isinstance(manifest.get("assets"), list) else []
            result.append(
                {
                    "task_id": str(data.get("task_id") or task_path.parent.name),
                    "project_id": project_id,
                    "title": str((_json(task_path.parent / "build_report.json").get("project") or {}).get("title") or project_id)
                    if (task_path.parent / "build_report.json").is_file()
                    else project_id,
                    "finished_at": str(data.get("finished_at") or data.get("updated_at") or ""),
                    "asset_count": len(assets),
                    "ai_count": sum(1 for item in assets if isinstance(item, dict) and item.get("ai_generated")),
                }
            )
        except (AssetReuseError, OSError, TypeError, ValueError):
            continue
    return sorted(result, key=lambda item: item["finished_at"], reverse=True)


def _candidate_copy(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "asset_id", "provider", "source_id", "title", "creator", "institution",
        "source_page", "download_url", "thumbnail_url", "rights_code", "rights_url",
        "width", "height", "mime", "selectable", "requires_reverification",
        "ai_generated", "semantic_status", "semantic_score", "semantic_review", "score",
        "score_detail", "generation", "perceptual_hash", "semantic_metadata",
    }
    return {key: value for key, value in candidate.items() if key in keep}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_reuse_source_snapshot(
    outputs_root: Path,
    cache_root: Path,
    parent_task_id: str,
) -> dict[str, Any]:
    run_dir = resolve_parent_task(outputs_root, parent_task_id)
    task = _json(run_dir / "task.json")
    scene_plan = _json(run_dir / "scene_plan.json")
    candidates = _json(run_dir / "asset_candidates.json")
    selection = _json(run_dir / "asset_selection.json")
    manifest = _json(run_dir / "assets_manifest.json")
    license_audit = _json(run_dir / "license_audit.json")
    if not manifest.get("human_reviewed") or not license_audit.get("asset_rights_ready"):
        raise AssetReuseError("父任务尚未完成人工素材审核或授权台账审计。")
    intents = {int(item["shot_id"]): item for item in scene_plan.get("shots", [])}
    manifest_by_shot = {int(item["shot_id"]): item for item in manifest.get("assets", [])}
    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_shots: dict[str, list[dict[str, Any]]] = {}
    for item in selection.get("selections", []):
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            continue
        candidate = item["candidate"]
        asset_id = str(candidate.get("asset_id") or "")
        if asset_id:
            selected_by_id[asset_id] = candidate
            selected_shots.setdefault(asset_id, []).append(item)

    pool: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_candidates = [
        candidate
        for shot in candidates.get("shots", [])
        for candidate in shot.get("candidates", [])
        if isinstance(candidate, dict)
        and (candidate.get("ai_generated") or str(candidate.get("asset_id")) in selected_by_id)
    ]
    for candidate in all_candidates:
        asset_id = str(candidate.get("asset_id") or "")
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        selected_usages = selected_shots.get(asset_id, [])
        selected_info = selected_usages[0] if selected_usages else None
        old_shot_id = int(selected_info.get("shot_id")) if selected_info else next(
            (
                int(shot.get("shot_id"))
                for shot in candidates.get("shots", [])
                if any(str(item.get("asset_id")) == asset_id for item in shot.get("candidates", []))
            ),
            0,
        )
        old_intent = intents.get(old_shot_id, {})
        manifest_item = next(
            (
                manifest_by_shot.get(int(usage.get("shot_id")))
                for usage in selected_usages
                if manifest_by_shot.get(int(usage.get("shot_id")))
                and str(manifest_by_shot[int(usage.get("shot_id"))].get("asset_id")) == asset_id
            ),
            None,
        )
        locator: dict[str, str] | None = None
        expected_sha = ""
        source_path: Path | None = None
        if manifest_item and str(manifest_item.get("asset_id")) == asset_id:
            expected_sha = str(manifest_item.get("sha256") or "")
            source_path = cache_root / "assets" / expected_sha
            locator = {"kind": "assets_sha256", "value": expected_sha}
        elif candidate.get("ai_generated"):
            preview = Path(str(candidate.get("local_preview") or ""))
            try:
                relative = preview.resolve().relative_to((cache_root / "ai").resolve())
            except (ValueError, OSError):
                relative = Path("")
            if relative and preview.is_file():
                source_path = preview
                expected_sha = _file_sha(preview)
                locator = {"kind": "ai_cache_file", "value": relative.as_posix()}
        if source_path is None or locator is None or not source_path.is_file():
            record = {"asset_id": asset_id, "selected": bool(selected_info), "reason": "cache_missing"}
            unavailable.append(record)
            if selected_info:
                raise AssetReuseError(f"父任务已选素材 {asset_id} 的内容缓存缺失，不能安全派生。")
            continue
        actual_sha = _file_sha(source_path)
        if expected_sha and actual_sha != expected_sha:
            raise AssetReuseError(f"父素材 {asset_id} 的缓存哈希已变化，已停止复用。")
        try:
            width, height, mime_extension = verify_image_bytes(
                source_path.read_bytes(), str(candidate.get("mime") or "image/jpeg"),
                min_long_edge=1, min_short_edge=1,
            )
        except (AssetSourceError, OSError) as exc:
            raise AssetReuseError(f"父素材 {asset_id} 已损坏，不能安全复用：{exc}") from exc
        pool.append(
            {
                "asset_id": asset_id,
                "selected_in_parent": bool(selected_usages),
                "parent_shot_id": old_shot_id,
                "parent_narration": str(old_intent.get("narration") or (selected_info or {}).get("narration") or ""),
                "parent_intent": old_intent,
                "parent_usages": [
                    {
                        "parent_shot_id": int(usage.get("shot_id")),
                        "parent_narration": str(
                            intents.get(int(usage.get("shot_id")), {}).get("narration")
                            or usage.get("narration")
                            or ""
                        ),
                        "parent_intent": intents.get(int(usage.get("shot_id")), {}),
                    }
                    for usage in selected_usages
                ]
                or [
                    {
                        "parent_shot_id": old_shot_id,
                        "parent_narration": str(old_intent.get("narration") or ""),
                        "parent_intent": old_intent,
                    }
                ],
                "candidate": _candidate_copy(candidate),
                "sha256": actual_sha,
                "cache_locator": locator,
                "width": width,
                "height": height,
                "verified_extension": mime_extension,
            }
        )
    if not pool:
        raise AssetReuseError("父任务没有可用的已选素材或 AI 候选缓存。")
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "scope": "selected_and_ai",
        "parent_task_id": str(task.get("task_id") or run_dir.name),
        "parent_project_id": str(task.get("project_id") or ""),
        "parent_input_hash": str(task.get("input_hash") or ""),
        "created_at": _now(),
        "asset_count": len(pool),
        "selected_count": sum(len(item.get("parent_usages", [])) for item in pool if item["selected_in_parent"]),
        "selected_asset_count": sum(1 for item in pool if item["selected_in_parent"]),
        "unused_ai_count": sum(1 for item in pool if item["candidate"].get("ai_generated") and not item["selected_in_parent"]),
        "unavailable": unavailable,
        "assets": pool,
    }
    snapshot["sha256"] = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return snapshot


def resolve_snapshot_asset(entry: dict[str, Any], cache_root: Path) -> Path:
    locator = entry.get("cache_locator") if isinstance(entry.get("cache_locator"), dict) else {}
    kind = str(locator.get("kind") or "")
    value = str(locator.get("value") or "")
    if kind == "assets_sha256" and re.fullmatch(r"[0-9a-f]{64}", value):
        path = cache_root / "assets" / value
    elif kind == "ai_cache_file" and value and not Path(value).is_absolute():
        path = cache_root / "ai" / Path(value)
    else:
        raise AssetReuseError(f"素材 {entry.get('asset_id')} 的缓存定位信息无效。")
    path = _inside(cache_root, path)
    if not path.is_file() or _file_sha(path) != str(entry.get("sha256") or ""):
        raise AssetReuseError(f"素材 {entry.get('asset_id')} 的缓存缺失或哈希变化。")
    return path


def _normalized(text: str) -> str:
    return "".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", str(text))).casefold()


def _terms(intent: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for name in ("must_include", "objects", "people", "search_terms_en", "search_terms_zh"):
        raw = intent.get(name)
        if isinstance(raw, list):
            values.extend(str(item).strip().casefold() for item in raw if str(item).strip())
    context = intent.get("time_context") if isinstance(intent.get("time_context"), dict) else {}
    values.extend(str(item).strip().casefold() for item in context.get("period_terms_en", []) if str(item).strip())
    return set(values)


def _local_match(new: dict[str, Any], old: dict[str, Any], order_distance: float) -> tuple[int, list[str], str]:
    new_text = _normalized(str(new.get("narration") or ""))
    old_text = _normalized(str(old.get("narration") or ""))
    if new_text and new_text == old_text:
        return 100, [], "旁白片段逐字一致。"
    ratio = SequenceMatcher(None, new_text, old_text).ratio() if new_text and old_text else 0.0
    new_terms = _terms(new)
    old_terms = _terms(old)
    overlap = len(new_terms & old_terms) / max(1, len(new_terms | old_terms))
    new_avoid = {str(item).casefold() for item in new.get("avoid", [])}
    old_positive = _terms(old)
    conflicts = sorted(item for item in new_avoid if any(item in value or value in item for value in old_positive))
    if conflicts:
        return 0, ["新镜头禁用元素与旧画面意图冲突：" + "、".join(conflicts)], "存在明确禁用冲突。"
    score = round(ratio * 58 + overlap * 27 + max(0.0, 1.0 - order_distance) * 15)
    return max(0, min(99, score)), [], f"原文相似度 {ratio:.0%}，意图词重合 {overlap:.0%}。"


def build_reuse_plan(
    scene_plan: dict[str, Any],
    snapshot: dict[str, Any],
    config: dict[str, Any],
    planner_config: dict[str, Any],
    env: dict[str, str],
    cache_root: Path,
) -> dict[str, Any]:
    assets = list(snapshot.get("assets", []))
    new_shots = list(scene_plan.get("shots", []))
    if not assets or not new_shots:
        raise AssetReuseError("复用匹配缺少新镜头或父素材。")
    parent_max = max((int(item.get("parent_shot_id") or 0) for item in assets), default=1)
    local_rows: list[dict[str, Any]] = []
    for new_index, shot in enumerate(new_shots, 1):
        rows = []
        for entry in assets:
            usages = entry.get("parent_usages") if isinstance(entry.get("parent_usages"), list) else []
            scored_usages = []
            for usage in usages or [entry]:
                old = usage.get("parent_intent") if isinstance(usage.get("parent_intent"), dict) else {}
                old = {**old, "narration": usage.get("parent_narration", "")}
                old_shot_id = int(usage.get("parent_shot_id") or entry.get("parent_shot_id") or 0)
                order_distance = abs(new_index / max(1, len(new_shots)) - old_shot_id / parent_max)
                score, conflicts, reason = _local_match(shot, old, order_distance)
                scored_usages.append((score, -old_shot_id, conflicts, reason, usage))
            score, _, conflicts, reason, best_usage = max(scored_usages, key=lambda item: (item[0], item[1]))
            rows.append(
                {
                    "asset_id": entry["asset_id"],
                    "parent_shot_id": best_usage.get("parent_shot_id", entry.get("parent_shot_id")),
                    "parent_narration": best_usage.get("parent_narration", entry.get("parent_narration", "")),
                    "selected_in_parent": bool(entry.get("selected_in_parent")),
                    "local_score": score,
                    "score": score,
                    "reason": reason,
                    "conflicts": conflicts,
                    "semantic_status": "exact" if score == 100 else "local_only",
                }
            )
        rows.sort(key=lambda item: (-int(item["score"]), not item["selected_in_parent"], int(item.get("parent_shot_id") or 0)))
        local_rows.append(
            {
                "shot_id": int(shot["shot_id"]),
                "narration": shot.get("narration", ""),
                "time_context": shot.get("time_context", {}),
                "must_include": shot.get("must_include", []),
                "avoid": shot.get("avoid", []),
                "candidates": rows[:8],
            }
        )

    semantic_status = "unavailable"
    safe_response: dict[str, Any] = {"batches": []}
    fuzzy = [
        {
            "shot_id": row["shot_id"],
            "new_intent": next(item for item in new_shots if int(item["shot_id"]) == row["shot_id"]),
            "candidates": [item for item in row["candidates"] if item["score"] < 100],
        }
        for row in local_rows
    ]
    secret_ref = str(planner_config.get("secret_ref") or "DEEPSEEK_API_KEY")
    api_key = env.get(secret_ref, "")
    if api_key and any(item["candidates"] for item in fuzzy):
        payload_rows = []
        by_asset = {str(item["asset_id"]): item for item in assets}
        for row in fuzzy:
            candidates_payload = []
            for candidate in row["candidates"]:
                entry = by_asset[str(candidate["asset_id"])]
                usage = next(
                    (
                        item
                        for item in entry.get("parent_usages", [])
                        if int(item.get("parent_shot_id") or 0) == int(candidate.get("parent_shot_id") or 0)
                    ),
                    entry,
                )
                candidates_payload.append(
                    {
                        "asset_id": candidate["asset_id"],
                        "old_narration": usage.get("parent_narration", ""),
                        "old_intent": usage.get("parent_intent", {}),
                        "local_score": candidate["local_score"],
                    }
                )
            payload_rows.append({"shot_id": row["shot_id"], "new_intent": row["new_intent"], "candidates": candidates_payload})
        successful_batches = 0
        batch_size = max(1, min(8, int(config.get("shots_per_batch", 4))))
        lookup: dict[tuple[int, str], dict[str, Any]] = {}
        for offset in range(0, len(payload_rows), batch_size):
            batch = payload_rows[offset : offset + batch_size]
            if not any(item.get("candidates") for item in batch):
                continue
            batch_payload = {"prompt_version": PROMPT_VERSION, "shots": batch}
            batch_key = hashlib.sha256(
                json.dumps(
                    {"payload": batch_payload, "planner": planner_config},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            batch_cache = cache_root / "visuals" / "reuse-semantic" / f"{batch_key}.json"
            try:
                if batch_cache.is_file():
                    cached = _json(batch_cache)
                    result = cached["result"]
                    response = {**cached.get("response", {}), "cache_hit": True}
                else:
                    result, response = request_json_object(
                        "你是历史短视频画面复用审校器。只比较文字元数据，不假装看过图片。"
                        "为每个候选返回0到100整数score、verdict(recommend|alternative|reject)、reason和conflicts。"
                        "时代或主题明确冲突必须reject。顶层只能是judgments数组。",
                        batch_payload,
                        planner_config,
                        api_key,
                    )
                    batch_cache.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_json(batch_cache, {"result": result, "response": response})
                judgments = result.get("judgments") if isinstance(result, dict) else None
                if not isinstance(judgments, list):
                    raise AssetReuseError("DeepSeek 复用匹配缺少 judgments 数组。")
                expected = {
                    (int(row["shot_id"]), str(candidate["asset_id"]))
                    for row in batch
                    for candidate in row.get("candidates", [])
                }
                batch_lookup = {
                    (int(item["shot_id"]), str(item["asset_id"])): item
                    for item in judgments
                    if isinstance(item, dict) and str(item.get("asset_id") or "")
                }
                if not expected.issubset(batch_lookup):
                    raise AssetReuseError("DeepSeek 复用匹配批次缺少候选判断。")
                lookup.update(batch_lookup)
                successful_batches += 1
                safe_response["batches"].append(
                    {"shot_ids": [int(item["shot_id"]) for item in batch], "status": "succeeded", **response}
                )
            except (DeepSeekPlannerError, AssetReuseError, KeyError, ValueError, TypeError) as exc:
                safe_response["batches"].append(
                    {"shot_ids": [int(item["shot_id"]) for item in batch], "status": "failed", "error": str(exc)}
                )
        for row in local_rows:
            for candidate in row["candidates"]:
                if candidate["score"] == 100:
                    continue
                judgment = lookup.get((row["shot_id"], str(candidate["asset_id"])))
                if not judgment:
                    continue
                try:
                    score = max(0, min(100, int(judgment.get("score"))))
                except (TypeError, ValueError):
                    continue
                conflicts = [str(item)[:300] for item in judgment.get("conflicts", [])] if isinstance(judgment.get("conflicts"), list) else []
                verdict = str(judgment.get("verdict") or "")
                if verdict == "reject" or conflicts:
                    score = min(score, int(config.get("alternative_threshold", 55)) - 1)
                candidate.update(
                    {
                        "score": score,
                        "reason": str(judgment.get("reason") or candidate["reason"])[:500],
                        "conflicts": conflicts,
                        "semantic_status": verdict if verdict in {"recommend", "alternative", "reject"} else "reviewed",
                    }
                )
        attempted_batches = len(safe_response["batches"])
        if attempted_batches and successful_batches == attempted_batches:
            semantic_status = "reviewed"
        elif successful_batches:
            semantic_status = "partial"

    recommendation_threshold = int(config.get("recommendation_threshold", 75))
    alternative_threshold = int(config.get("alternative_threshold", 55))
    used: set[str] = set()
    for row in local_rows:
        row["candidates"].sort(key=lambda item: (-int(item["score"]), not item["selected_in_parent"], str(item["asset_id"])))
        row["candidates"] = [item for item in row["candidates"] if int(item["score"]) >= alternative_threshold or item["score"] == 100]
        recommendation = None
        for item in row["candidates"]:
            exact = item["score"] == 100
            can_auto = exact or semantic_status == "reviewed"
            if can_auto and int(item["score"]) >= recommendation_threshold and not item["conflicts"] and item["asset_id"] not in used:
                recommendation = item["asset_id"]
                used.add(str(item["asset_id"]))
                break
        row["recommended_asset_id"] = recommendation
    plan = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "parent_task_id": snapshot.get("parent_task_id"),
        "parent_project_id": snapshot.get("parent_project_id"),
        "snapshot_sha256": snapshot.get("sha256"),
        "semantic_status": semantic_status,
        "recommendation_threshold": recommendation_threshold,
        "alternative_threshold": alternative_threshold,
        "deepseek_response": safe_response,
        "shots": local_rows,
    }
    plan["sha256"] = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return plan


def _validate_reuse_selection(
    plan: dict[str, Any],
    actions: dict[int, str],
    duplicate_overrides: set[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    uses: dict[str, list[int]] = {}
    for shot in plan.get("shots", []):
        shot_id = int(shot["shot_id"])
        action = actions.get(shot_id)
        if not action:
            raise AssetReuseError(f"镜头 {shot_id} 尚未明确选择复用旧图或补充新图。")
        if action == "__new__":
            selections.append({"shot_id": shot_id, "action": "new"})
            continue
        lookup = {str(item["asset_id"]): item for item in shot.get("candidates", [])}
        candidate = lookup.get(action)
        if not candidate:
            raise AssetReuseError(f"镜头 {shot_id} 提交了不在复用清单中的素材。")
        if candidate.get("conflicts"):
            raise AssetReuseError(f"镜头 {shot_id} 的候选存在明确时代或主题冲突，不能复用。")
        uses.setdefault(action, []).append(shot_id)
        selections.append(
            {
                "shot_id": shot_id,
                "action": "reuse",
                "asset_id": action,
                "parent_shot_id": candidate.get("parent_shot_id"),
                "reuse_score": int(candidate.get("score", 0)),
                "reuse_reason": str(candidate.get("reason") or ""),
                "duplicate_override": shot_id in duplicate_overrides,
            }
        )
    if int(config.get("max_uses_per_asset", 1)) == 1:
        for asset_id, shot_ids in uses.items():
            if len(shot_ids) > 1 and not all(shot_id in duplicate_overrides for shot_id in shot_ids[1:]):
                raise AssetReuseError(
                    f"旧素材 {asset_id} 被多个镜头复用；从第二次使用开始必须勾选重复使用确认。"
                )
    return {
        "schema_version": 1,
        "reviewed": True,
        "reviewed_at": _now(),
        "parent_task_id": plan.get("parent_task_id"),
        "selections": selections,
    }


def run_reuse_review_server(
    run_dir: Path,
    plan: dict[str, Any],
    snapshot: dict[str, Any],
    config: dict[str, Any],
    cache_root: Path,
    *,
    open_browser: bool = True,
) -> str:
    entries = {str(item["asset_id"]): item for item in snapshot.get("assets", [])}
    previews = {asset_id: resolve_snapshot_asset(entry, cache_root) for asset_id, entry in entries.items()}
    csrf = secrets.token_urlsafe(24)
    holder: dict[str, ThreadingHTTPServer] = {}
    status = {"value": "waiting"}

    def page(message: str = "") -> str:
        sections = []
        warning = "" if plan.get("semantic_status") == "reviewed" else '<div class="warn">DeepSeek 不可用：仅逐字匹配会预选，其他候选请人工查看。</div>'
        for shot in plan.get("shots", []):
            cards = []
            for candidate in shot.get("candidates", []):
                asset_id = str(candidate["asset_id"])
                entry = entries[asset_id]
                checked = "checked" if shot.get("recommended_asset_id") == asset_id else ""
                badge = "父成片已选" if entry.get("selected_in_parent") else "父任务未选 AI"
                cards.append(
                    f'<label class="card"><input type="radio" name="shot_{shot["shot_id"]}" value="{html.escape(asset_id)}" {checked}>'
                    f'<img src="/thumb/{urllib.parse.quote(asset_id)}"><b>{html.escape(str(entry["candidate"].get("title") or asset_id))}</b>'
                    f'<span>{badge} · 分数 {int(candidate.get("score",0))}</span>'
                    f'<span>{html.escape(str(entry["candidate"].get("provider") or ""))} · '
                    f'{html.escape(str(entry["candidate"].get("rights_code") or "未核验"))} · '
                    f'{int(entry.get("width") or 0)}×{int(entry.get("height") or 0)}</span>'
                    f'<span>旧旁白：{html.escape(str(candidate.get("parent_narration") or ""))}</span>'
                    f'<span>{html.escape(str(candidate.get("reason") or ""))}</span>'
                    f'<span class="err">{html.escape("；".join(str(item) for item in candidate.get("conflicts", [])))}</span>'
                    f'<a href="{html.escape(str(entry["candidate"].get("source_page") or "#"), quote=True)}" target="_blank" rel="noreferrer">来源页</a></label>'
                )
            context = shot.get("time_context") if isinstance(shot.get("time_context"), dict) else {}
            sections.append(
                f'<section><h2>镜头 {int(shot["shot_id"])}</h2><p>新旁白：{html.escape(str(shot.get("narration") or ""))}</p>'
                f'<p><span>时代：{html.escape(str(context.get("label") or "未知"))} · '
                f'{html.escape(str(context.get("region") or "未知"))}<br>必须出现：'
                f'{html.escape("；".join(str(item) for item in shot.get("must_include", [])) or "无")}<br>避免出现：'
                f'{html.escape("；".join(str(item) for item in shot.get("avoid", [])) or "无")}</span></p>'
                f'<div class="grid">{"".join(cards)}</div>'
                f'<label class="new"><input type="radio" name="shot_{int(shot["shot_id"])}" value="__new__" '
                f'{"checked" if not shot.get("recommended_asset_id") else ""}> 补充新图（之后才会搜索或生图）</label>'
                f'<label class="duplicate"><input type="checkbox" name="duplicate_{int(shot["shot_id"])}" value="1"> '
                '若重复使用同一旧图，我已确认视觉重复风险</label></section>'
            )
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>旧画面匹配审核</title><style>body{{margin:0;background:#111;color:#f5f5f2;font:15px/1.55 system-ui,"Microsoft YaHei"}}main{{max-width:1200px;margin:auto;padding:26px}}header{{position:sticky;top:0;background:#111e;padding:8px 0;z-index:3}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{display:flex;flex-direction:column;gap:6px;background:#1d1f23;padding:10px;border:2px solid transparent;border-radius:12px}}.card:has(input:checked){{border-color:#ffd54a}}img{{width:100%;aspect-ratio:9/14;object-fit:cover}}section{{margin:24px 0 38px}}span{{color:#bbb}}.new,.duplicate{{display:block;margin-top:12px;padding:10px;background:#202329}}button{{padding:13px 22px;background:#ffd54a;border:0;border-radius:9px;font-weight:800}}.warn{{background:#4a3d20;padding:10px}}.err{{color:#ff8d8d}}</style></head><body><main><header><h1>旧画面匹配审核</h1><p>每个镜头必须明确选择旧图或补充新图。锁定后，只有“补充新图”的镜头会调用搜索和 ComfyUI。</p>{warning}<div class="err">{html.escape(message)}</div></header><form method="post" action="/submit"><input type="hidden" name="csrf" value="{csrf}">{''.join(sections)}<button>确认复用选择并继续</button></form></main></body></html>'''

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self.send(200, page().encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path.startswith("/thumb/"):
                asset_id = urllib.parse.unquote(self.path.removeprefix("/thumb/"))
                path = previews.get(asset_id)
                if path:
                    self.send(200, path.read_bytes(), "image/jpeg")
                else:
                    self.send(404, b"", "text/plain")
                return
            self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/submit":
                self.send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length < 0 or length > 128_000:
                self.send(413, b"request too large", "text/plain")
                return
            values = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            if values.get("csrf", [""])[0] != csrf:
                self.send(403, b"forbidden", "text/plain")
                return
            actions = {int(key.removeprefix("shot_")): value[0] for key, value in values.items() if key.startswith("shot_") and value}
            overrides = {int(key.removeprefix("duplicate_")) for key, value in values.items() if key.startswith("duplicate_") and value and value[0] == "1"}
            try:
                selection = _validate_reuse_selection(plan, actions, overrides, config)
            except AssetReuseError as exc:
                self.send(400, page(str(exc)).encode("utf-8"), "text/html; charset=utf-8")
                return
            _atomic_json(run_dir / "asset_reuse_selection.json", selection)
            reused = [item for item in selection["selections"] if item["action"] == "reuse"]
            _atomic_json(
                run_dir / "reuse_report.json",
                {
                    "schema_version": 1,
                    "parent_task_id": plan.get("parent_task_id"),
                    "total_shots": len(selection["selections"]),
                    "reused_shots": len(reused),
                    "new_shots": len(selection["selections"]) - len(reused),
                    "duplicate_overrides": sum(1 for item in reused if item.get("duplicate_override")),
                    "reviewed_at": selection["reviewed_at"],
                },
            )
            status["value"] = "submitted"
            self.send(200, '<meta charset="utf-8"><h2>复用审核已提交，可以关闭页面。</h2>'.encode("utf-8"), "text/html; charset=utf-8")
            threading.Thread(target=holder["server"].shutdown, daemon=True).start()

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    holder["server"] = server
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[画面复用审核] 仅本机可访问：{url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return str(status["value"])


def unmatched_scene_plan(scene_plan: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    actions = {int(item["shot_id"]): item.get("action") for item in selection.get("selections", [])}
    return {"shots": [item for item in scene_plan.get("shots", []) if actions.get(int(item["shot_id"])) == "new"]}


def merge_reused_candidates(
    scene_plan: dict[str, Any],
    fresh: dict[str, Any],
    selection: dict[str, Any],
    snapshot: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    fresh_by_shot = {int(item["shot_id"]): item for item in fresh.get("shots", [])}
    selected = {int(item["shot_id"]): item for item in selection.get("selections", [])}
    entries = {str(item["asset_id"]): item for item in snapshot.get("assets", [])}
    shots: list[dict[str, Any]] = []
    for intent in scene_plan.get("shots", []):
        shot_id = int(intent["shot_id"])
        choice = selected.get(shot_id)
        if not choice:
            raise AssetReuseError(f"镜头 {shot_id} 缺少复用审核选择。")
        if choice.get("action") == "new":
            if shot_id not in fresh_by_shot:
                raise AssetReuseError(f"镜头 {shot_id} 缺少待补充候选清单。")
            shots.append(fresh_by_shot[shot_id])
            continue
        entry = entries.get(str(choice.get("asset_id") or ""))
        if not entry:
            raise AssetReuseError(f"镜头 {shot_id} 引用了不在父快照中的素材。")
        candidate = dict(entry["candidate"])
        candidate.update(
            {
                "local_preview": str(resolve_snapshot_asset(entry, cache_root)),
                "selectable": True,
                "reused": True,
                "reused_from": {
                    "parent_task_id": snapshot.get("parent_task_id"),
                    "parent_project_id": snapshot.get("parent_project_id"),
                    "parent_shot_id": choice.get("parent_shot_id", entry.get("parent_shot_id")),
                    "asset_id": entry.get("asset_id"),
                    "sha256": entry.get("sha256"),
                },
                "reuse_score": choice.get("reuse_score"),
                "reuse_reason": choice.get("reuse_reason"),
                "duplicate_override": bool(choice.get("duplicate_override")),
                "reuse_cache_locator": entry.get("cache_locator"),
                "reuse_sha256": entry.get("sha256"),
            }
        )
        shots.append(
            {
                "shot_id": shot_id,
                "intent_id": intent["intent_id"],
                "start": intent.get("start", 0.0),
                "end": intent.get("end", 0.0),
                "narration": intent.get("narration", ""),
                "query": "",
                "successful_providers": [candidate.get("provider")],
                "museum_source_succeeded": not candidate.get("ai_generated"),
                "recommended_asset_id": candidate["asset_id"],
                "candidates": [candidate],
                "time_context": intent.get("time_context", {}),
                "must_include": intent.get("must_include", []),
                "avoid": intent.get("avoid", []),
                "semantic_review_status": "reused_human_reviewed",
            }
        )
    merged = dict(fresh)
    merged["shots"] = shots
    merged["reuse_enabled"] = True
    merged["reused_shot_count"] = sum(1 for item in selection.get("selections", []) if item.get("action") == "reuse")
    merged["new_shot_count"] = len(shots) - int(merged["reused_shot_count"])
    return merged
