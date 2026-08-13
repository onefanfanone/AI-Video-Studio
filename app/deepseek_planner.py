from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .scene_plan_schema import SCENE_SCHEMA, SHOT_REQUIRED, validate_plan_shots


class DeepSeekPlannerError(RuntimeError):
    """A redacted, user-facing error from the DeepSeek planning layer."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Video/3.1",
        },
        method="POST",
    )
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            time.sleep(float(attempt + 1))
    raise DeepSeekPlannerError(f"DeepSeek API 请求失败（已重试两次）：{last_error}")


def request_json_object(
    system_prompt: str,
    user_payload: dict[str, Any],
    settings: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise DeepSeekPlannerError("DeepSeek JSON 复核需要 DEEPSEEK_API_KEY。")
    model = str(settings.get("model", "deepseek-v4-flash"))
    endpoint = str(settings.get("base_url", "https://api.deepseek.com")).rstrip(
        "/"
    ) + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": int(settings.get("max_tokens", 12000)),
    }
    thinking = str(settings.get("thinking", "disabled")).lower()
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    raw = _post_json(
        endpoint,
        payload,
        api_key,
        timeout=float(settings.get("timeout_seconds", 120)),
    )
    choices = raw.get("choices") or []
    if not choices:
        raise DeepSeekPlannerError("DeepSeek JSON 响应没有 choices。")
    choice = choices[0]
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason != "stop":
        raise DeepSeekPlannerError(
            f"DeepSeek JSON 响应未正常结束：finish_reason={finish_reason or 'missing'}。"
        )
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise DeepSeekPlannerError("DeepSeek JSON 响应的 message.content 不是字符串。")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DeepSeekPlannerError("DeepSeek JSON 响应不是有效 JSON。") from exc
    safe = {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "created": raw.get("created"),
        "usage": raw.get("usage"),
        "system_fingerprint": raw.get("system_fingerprint"),
        "finish_reason": finish_reason,
    }
    return result, safe


def _validate_plan(
    plan: Any, expected_shots: int, historical_consistency: str = "off"
) -> list[dict[str, Any]]:
    # historical_consistency remains an accepted compatibility argument, but
    # scene-plan-v3 derives every constraint from the shot itself.
    return validate_plan_shots(plan, expected_shots)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_attempt_diagnostics(
    path: Path | None, attempts: list[dict[str, Any]]
) -> None:
    if path is not None:
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "provider": "deepseek",
                "kind": "scene_plan_attempts",
                "attempts": attempts,
            },
        )


def _safe_response(raw: dict[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "created": raw.get("created"),
        "usage": raw.get("usage"),
        "system_fingerprint": raw.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
    }


def _decode_plan_response(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    safe = _safe_response(raw)
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("响应没有 choices。")
    choice = choices[0]
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason != "stop":
        raise ValueError(f"finish_reason={finish_reason or 'missing'}。")
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise ValueError("message.content 不是字符串。")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("message.content 不是有效 JSON。") from exc
    if not isinstance(result, dict):
        raise ValueError("JSON 顶层不是对象。")
    return result, safe


def _normalize_search_anchors(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Add a deterministic museum query using facts already present in each shot."""
    changes: list[dict[str, Any]] = []
    shots = plan.get("shots")
    if not isinstance(shots, list):
        return changes
    for position, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            continue
        context = shot.get("time_context")
        if not isinstance(context, dict):
            continue
        periods = context.get("period_terms_en")
        period = ""
        if isinstance(periods, list):
            period = next(
                (str(item).strip() for item in periods if str(item).strip()), ""
            )
        period = period or str(context.get("label") or "").strip()
        subjects: list[str] = []
        for field in ("must_include", "objects"):
            values = shot.get(field)
            if isinstance(values, list):
                subjects.extend(str(item).strip() for item in values if str(item).strip())
        if not subjects and str(shot.get("action") or "").strip():
            subjects.append(str(shot["action"]).strip())
        queries = shot.get("search_terms_en")
        if not period or not subjects or not isinstance(queries, list):
            continue
        canonical = f"{period} {subjects[0]} museum collection"
        if any(str(item).strip().lower() == canonical.lower() for item in queries):
            continue
        shot["search_terms_en"] = [canonical, *queries]
        changes.append(
            {
                "shot_id": int(shot.get("shot_id") or position),
                "field": "search_terms_en",
                "added": canonical,
            }
        )
    return changes


def _collect_shot_errors(
    plan: Any, expected_shots: int
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    if not isinstance(plan, dict) or set(plan) != {"shots"}:
        raise ValueError("顶层必须只包含 shots。")
    shots = plan.get("shots")
    if not isinstance(shots, list) or len(shots) != expected_shots:
        actual = len(shots) if isinstance(shots, list) else 0
        raise ValueError(f"镜头数必须为 {expected_shots}（本次返回 {actual}）。")
    errors: dict[int, str] = {}
    for position, shot in enumerate(shots, 1):
        try:
            validate_plan_shots({"shots": [shot]}, 1)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("镜头 1"):
                message = f"镜头 {position}" + message[len("镜头 1") :]
            errors[position] = message
    return [dict(item) if isinstance(item, dict) else item for item in shots], errors


def audit_scene_plan(
    script: str,
    scene_plan: dict[str, Any],
    planner: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_config = dict(planner.get("semantic_audit", {}))
    if not audit_config.get("enabled", True):
        result = dict(scene_plan)
        result["semantic_audit_status"] = "disabled"
        return result, {"status": "disabled", "attempts": []}
    settings = {**planner, **audit_config}
    maximum_repairs = int(audit_config.get("max_repair_rounds", 2))
    if maximum_repairs < 0 or maximum_repairs > 4:
        raise DeepSeekPlannerError("max_repair_rounds 必须在 0 到 4 之间。")
    runtime = {
        int(shot["shot_id"]): {
            key: shot.get(key) for key in ("start", "end", "narration")
        }
        for shot in scene_plan.get("shots", [])
    }
    current = [
        {key: shot[key] for key in SHOT_REQUIRED}
        for shot in scene_plan.get("shots", [])
    ]
    attempts: list[dict[str, Any]] = []
    system_prompt = (
        "你是历史短视频视觉计划的语义审校员。只根据用户提供的文案和计划判断内部一致性，"
        "不得增加史实断言。检查每个镜头的 time_context、人物、物件、动作、搜索词、"
        "must_include、avoid 和 ai_prompt 是否相互一致，尤其不能把抽象原理改画成不属于"
        "目标时代的场景。你必须输出 JSON，顶层只能有 valid、issues、shots。issues 是"
        "{shot_id, conflicts[]} 数组；shots 必须始终返回完整、已修订的 scene-plan-v3 镜头数组。"
        "如果没有问题，valid=true、issues=[]，shots 原样返回。"
    )
    for audit_round in range(maximum_repairs + 1):
        response, safe = request_json_object(
            system_prompt,
            {
                "script": script,
                "scene_schema": SCENE_SCHEMA,
                "shots": current,
                "repair_round": audit_round,
            },
            settings,
            api_key,
        )
        if not isinstance(response, dict) or set(response) != {"valid", "issues", "shots"}:
            raise DeepSeekPlannerError("DeepSeek 计划审校响应字段不符合约定。")
        if not isinstance(response.get("valid"), bool) or not isinstance(
            response.get("issues"), list
        ):
            raise DeepSeekPlannerError("DeepSeek 计划审校的 valid/issues 类型不正确。")
        try:
            revised = validate_plan_shots(
                {"shots": response.get("shots")}, len(current)
            )
        except ValueError as exc:
            raise DeepSeekPlannerError(
                f"DeepSeek 计划审校返回的修订计划无效：{exc}"
            ) from exc
        issue_rows: list[dict[str, Any]] = []
        for issue in response["issues"]:
            if not isinstance(issue, dict) or set(issue) != {"shot_id", "conflicts"}:
                raise DeepSeekPlannerError("DeepSeek 计划审校 issues 字段不符合约定。")
            if not isinstance(issue.get("shot_id"), int) or not isinstance(
                issue.get("conflicts"), list
            ) or any(not isinstance(item, str) for item in issue["conflicts"]):
                raise DeepSeekPlannerError("DeepSeek 计划审校 issue 类型不正确。")
            issue_rows.append(
                {
                    "shot_id": int(issue["shot_id"]),
                    "conflicts": list(issue["conflicts"]),
                }
            )
        attempts.append(
            {
                "round": audit_round,
                "valid": response["valid"],
                "issues": issue_rows,
                "response": safe,
            }
        )
        current = revised
        if response["valid"] and not issue_rows:
            merged: list[dict[str, Any]] = []
            for shot in revised:
                shot_id = int(shot["shot_id"])
                merged.append({**shot, **runtime.get(shot_id, {})})
            result = {**scene_plan, "schema_version": 2, "shots": merged}
            result["semantic_audit_status"] = "reviewed"
            return result, {"status": "reviewed", "attempts": attempts}
        if audit_round < maximum_repairs:
            print(
                f"      计划语义审校发现 {len(issue_rows)} 个镜头问题，"
                f"正在自动修订（{audit_round + 1}/{maximum_repairs}）..."
            )
    detail = "; ".join(
        conflict
        for issue in attempts[-1].get("issues", [])
        for conflict in issue.get("conflicts", [])
    ) or "审校仍报告未解决的内部冲突。"
    raise DeepSeekPlannerError(f"视觉计划自动修订后仍未通过语义审校：{detail}")


def create_scene_plan(
    script: str,
    shots: list[dict[str, Any]],
    planner: dict[str, Any],
    api_key: str,
    diagnostics_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise DeepSeekPlannerError(
            "DeepSeek 场景规划需要 DEEPSEEK_API_KEY。请把它写入项目根目录 .env.local；"
            "密钥不会进入任务文件或报告。"
        )
    shot_inputs = [
        {
            "shot_id": index + 1,
            "start": shot["start"],
            "end": shot["end"],
            "narration": shot.get("caption_text", ""),
        }
        for index, shot in enumerate(shots)
    ]
    system_prompt = (
        "你是历史短视频的画面资料编辑。你必须只输出一个符合用户所给 JSON Schema 的 JSON 对象，"
        "不能输出 Markdown、解释或代码围栏。只把给定文案和整数帧镜头转换为检索意图；不得改写"
        "文案，不得补充文案没有陈述的历史事实或断言。每个镜头必须对应一个 intent，并生成"
        "time_context：时代标签、地区、可为空的起止年份、置信度和中英文时代锚点。must_include"
        "和 avoid 必须根据当前镜头动态生成；avoid 要写具体可见的时代错置物，不得只写笼统的"
        "modern objects。搜索词必须同时包含时代/文化锚点和具体物件、人物或工艺，不能只搜索"
        "抽象概念。ai_prompt 必须是英文竖屏电影感写实重构，明确年代、材质、服饰、构图和底部"
        "字幕安全区。文字、水印、Logo 等通用限制由渲染层追加，不要用固定文明词表代替判断。"
    )
    user_prompt = json.dumps(
        {
            "instruction": "Return JSON only. Follow scene_schema exactly.",
            "scene_schema": SCENE_SCHEMA,
            "script": script,
            "shots": shot_inputs,
        },
        ensure_ascii=False,
    )
    model = str(planner.get("model", "deepseek-v4-flash"))
    base_url = str(planner.get("base_url", "https://api.deepseek.com")).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": int(planner.get("max_tokens", 12000)),
    }
    thinking = str(planner.get("thinking", "disabled")).lower()
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}

    attempts: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    normalizations: list[dict[str, Any]] = []
    last_safe: dict[str, Any] = {}
    base_plan: dict[str, Any] | None = None
    shot_errors: dict[int, str] = {}
    for attempt in range(3):
        full_payload = dict(payload)
        if invalid_reasons:
            retry_request = {
                "instruction": (
                    "Generate a fresh complete plan. Return exactly one shot for every input "
                    f"shot, preserving all {len(shots)} positions. Do not return a partial patch."
                ),
                "scene_schema": SCENE_SCHEMA,
                "script": script,
                "shots": shot_inputs,
                "previous_validation_error": invalid_reasons[-1],
            }
            full_payload["messages"] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(retry_request, ensure_ascii=False),
                },
            ]
        raw = _post_json(
            endpoint,
            full_payload,
            api_key,
            timeout=float(planner.get("timeout_seconds", 120)),
        )
        safe = _safe_response(raw)
        last_safe = safe
        record: dict[str, Any] = {
            "phase": "initial_plan",
            "round": attempt,
            "response": safe,
        }
        try:
            decoded, _ = _decode_plan_response(raw)
            rows = decoded.get("shots")
            record["shot_count"] = len(rows) if isinstance(rows, list) else 0
            changes = _normalize_search_anchors(decoded)
            candidate_shots, errors = _collect_shot_errors(decoded, len(shots))
            normalizations = list(changes)
            base_plan = {"shots": candidate_shots}
            shot_errors = errors
            record["normalizations"] = changes
            record["shot_errors"] = [
                {"shot_id": shot_id, "error": error}
                for shot_id, error in sorted(errors.items())
            ]
            attempts.append(record)
            _write_attempt_diagnostics(diagnostics_path, attempts)
            break
        except ValueError as exc:
            invalid_reasons.append(str(exc))
            record["error"] = str(exc)
            attempts.append(record)
            _write_attempt_diagnostics(diagnostics_path, attempts)
            if attempt < 2:
                print(
                    "      DeepSeek 返回的整份计划结构不完整，正在无历史截断地重新生成 "
                    f"（重试 {attempt + 1}/2）：{exc}"
                )
    if base_plan is None:
        detail = invalid_reasons[-1] if invalid_reasons else "未知结构错误。"
        raise DeepSeekPlannerError(
            f"DeepSeek 视觉计划连续三次未返回完整 {len(shots)} 镜头：{detail}"
        )

    if normalizations:
        print(
            f"      已用镜头自身的时代和必需元素补全 {len(normalizations)} 条馆藏搜索锚点。"
        )

    if shot_errors:
        maximum_local_repairs = int(planner.get("max_local_repair_rounds", 10))
        if not 1 <= maximum_local_repairs <= 10:
            raise DeepSeekPlannerError(
                "max_local_repair_rounds 必须是 1 到 10 之间的整数。"
            )
        repair_system = (
            "你是历史短视频视觉计划的局部修订器。只修订请求中列出的镜头，不得返回其他镜头，"
            "不得改写旁白或增加旁白未陈述的事实。必须返回 JSON 对象，顶层只能有 shots；shots "
            "必须包含且只包含 requested_shot_ids，对每个 ID 返回完整 scene-plan-v3 镜头对象，而不是"
            "字段补丁。搜索词同时包含 time_context 的时代/文化锚点和 must_include/objects 的具体"
            "物件或工艺。"
        )
        item_schema = SCENE_SCHEMA["properties"]["shots"]["items"]
        for repair_round in range(maximum_local_repairs):
            invalid_ids = sorted(shot_errors)
            print(
                f"      计划中有 {len(invalid_ids)} 个镜头需要局部修正："
                + "、".join(str(item) for item in invalid_ids)
                + f"（{repair_round + 1}/{maximum_local_repairs}）"
            )
            repair_payload = {
                "instruction": "Return only the corrected requested shots as JSON.",
                "requested_shot_ids": invalid_ids,
                "shot_schema": item_schema,
                "script": script,
                "invalid_shots": [
                    {
                        "shot_id": shot_id,
                        "timeline": shot_inputs[shot_id - 1],
                        "validation_error": shot_errors[shot_id],
                        "current_shot": base_plan["shots"][shot_id - 1],
                    }
                    for shot_id in invalid_ids
                ],
            }
            try:
                repaired, safe = request_json_object(
                    repair_system,
                    repair_payload,
                    {
                        **planner,
                        "max_tokens": min(int(planner.get("max_tokens", 12000)), 6000),
                    },
                    api_key,
                )
                last_safe = safe
                rows = repaired.get("shots") if isinstance(repaired, dict) else None
                returned_ids = (
                    [int(item.get("shot_id", -1)) for item in rows if isinstance(item, dict)]
                    if isinstance(rows, list)
                    else []
                )
                if not isinstance(repaired, dict) or set(repaired) != {"shots"}:
                    raise ValueError("局部修订顶层必须只包含 shots。")
                if not isinstance(rows, list) or len(rows) != len(invalid_ids):
                    actual = len(rows) if isinstance(rows, list) else 0
                    raise ValueError(
                        f"局部修订应返回 {len(invalid_ids)} 个镜头，本次返回 {actual}。"
                    )
                if len(set(returned_ids)) != len(returned_ids) or sorted(returned_ids) != invalid_ids:
                    raise ValueError(
                        "局部修订返回的 shot_id 必须与 requested_shot_ids 完全一致。"
                    )
                by_id = {int(item["shot_id"]): dict(item) for item in rows}
                for shot_id in invalid_ids:
                    base_plan["shots"][shot_id - 1] = by_id[shot_id]
                changes = _normalize_search_anchors(base_plan)
                normalizations.extend(changes)
                candidate_shots, shot_errors = _collect_shot_errors(
                    base_plan, len(shots)
                )
                base_plan = {"shots": candidate_shots}
                attempts.append(
                    {
                        "phase": "targeted_repair",
                        "round": repair_round,
                        "requested_shot_ids": invalid_ids,
                        "returned_shot_ids": returned_ids,
                        "response": safe,
                        "normalizations": changes,
                        "shot_errors": [
                            {"shot_id": shot_id, "error": error}
                            for shot_id, error in sorted(shot_errors.items())
                        ],
                    }
                )
                _write_attempt_diagnostics(diagnostics_path, attempts)
                if not shot_errors:
                    break
            except (DeepSeekPlannerError, ValueError, TypeError) as exc:
                attempts.append(
                    {
                        "phase": "targeted_repair",
                        "round": repair_round,
                        "requested_shot_ids": invalid_ids,
                        "response": last_safe,
                        "error": str(exc),
                    }
                )
                _write_attempt_diagnostics(diagnostics_path, attempts)
                if isinstance(exc, DeepSeekPlannerError):
                    raise
        if shot_errors:
            detail = "；".join(
                shot_errors[shot_id] for shot_id in sorted(shot_errors)
            )
            raise DeepSeekPlannerError(
                f"DeepSeek 局部修订 {maximum_local_repairs} 轮后仍有 "
                f"{len(shot_errors)} 个镜头未通过：{detail}"
            )

    intents = validate_plan_shots(base_plan, len(shots))

    for index, intent in enumerate(intents, 1):
        intent["shot_id"] = index
        intent["intent_id"] = f"intent-{index:03d}"
        intent["start"] = shots[index - 1]["start"]
        intent["end"] = shots[index - 1]["end"]
        intent["narration"] = shots[index - 1].get("caption_text", "")
    result = {
        "schema_version": 2,
        "provider": "deepseek",
        "model": model,
        "prompt_version": planner.get("prompt_version", "scene-plan-v3"),
        "response_id": last_safe.get("id"),
        "local_normalizations": normalizations,
        "shots": intents,
    }
    safe_raw = {
        **last_safe,
        "validation_attempts": len(attempts),
        "attempts": attempts,
    }
    _write_attempt_diagnostics(diagnostics_path, attempts)
    return result, safe_raw
