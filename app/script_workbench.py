from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import shutil
import threading
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import yaml

from .deepseek_planner import DeepSeekPlannerError, request_json_object
from .asset_reuse import (
    AssetReuseError,
    create_reuse_source_snapshot,
    resolve_parent_task,
)
from .openai_visuals import load_local_env
from .studio_profiles import ProfileError, ProfileStore, snapshot_hash
from .studio_settings import SecretStore, SettingsStore, discover_jianying_draft_root, get_studio_paths


ROOT = Path(__file__).resolve().parents[1]
_PATHS = get_studio_paths(ROOT)
DRAFTS_ROOT = _PATHS.draft_root
TEMPLATE_PATH = ROOT / "config" / "project_template.yaml"
VOICE_PROFILES_PATH = ROOT / "config" / "voice_profiles.yaml"
PROMPT_VERSION = "script-workbench-v1"
ALLOWED_MODES = {"direct", "review", "topic"}
ALLOWED_DURATIONS = {30, 45, 60, 90}
ALLOWED_VISUAL_STRATEGIES = {"museum_and_ai", "ai_only", "local"}
MAX_AI_VERSIONS = 10
MAX_REQUEST_BYTES = 256_000


class ScriptWorkbenchError(RuntimeError):
    """A recoverable script-workbench error safe to show in the local UI."""


def configure_workbench_paths() -> None:
    """Refresh mutable paths after first-run setup changes the active workspace."""
    global DRAFTS_ROOT
    DRAFTS_ROOT = get_studio_paths(ROOT).draft_root


def load_voice_profiles() -> dict[str, dict[str, str]]:
    """Load the user-facing profile catalog instead of hard-coding voices in HTML."""
    try:
        payload = yaml.safe_load(VOICE_PROFILES_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScriptWorkbenchError(f"找不到配音 profile 配置：{VOICE_PROFILES_PATH}") from exc
    profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
    if not isinstance(profiles, dict) or not profiles:
        raise ScriptWorkbenchError("配音 profile 配置为空。")
    result: dict[str, dict[str, str]] = {}
    for profile_id, raw in profiles.items():
        if not re.fullmatch(r"[a-z0-9_]+", str(profile_id)) or not isinstance(raw, dict):
            continue
        voice = str(raw.get("voice") or "").strip()
        if not voice:
            continue
        result[str(profile_id)] = {
            "label": str(raw.get("label") or profile_id).strip()[:80],
            "voice": voice[:100],
            "rate": str(raw.get("rate") or "+0%").strip()[:16],
            "pitch": str(raw.get("pitch") or "+0Hz").strip()[:16],
        }
    if not result:
        raise ScriptWorkbenchError("配音 profile 配置没有可用项。")
    return result


def default_voice_profile() -> str:
    profiles = load_voice_profiles()
    try:
        template = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
        selected = str(template.get("voice", {}).get("profile") or "")
    except (OSError, AttributeError):
        selected = ""
    return selected if selected in profiles else next(iter(profiles))


def voice_preview_files() -> dict[str, Path]:
    """Return only catalogued MP3s declared by the newest audition manifest."""
    outputs = get_studio_paths(ROOT).output_root
    manifests = sorted(
        outputs.glob("voice-audition-*/comparison.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if outputs.is_dir() else []
    profiles = load_voice_profiles()
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result: dict[str, Path] = {}
        for sample in payload.get("samples", []):
            if not isinstance(sample, dict):
                continue
            profile = str(sample.get("profile") or "")
            filename = str(sample.get("file") or "")
            if profile not in profiles or Path(filename).name != filename:
                continue
            candidate = (manifest_path.parent / filename).resolve()
            try:
                candidate.relative_to(manifest_path.parent.resolve())
            except ValueError:
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".mp3":
                result[profile] = candidate
        if result:
            return result
    return {}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_script(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def script_stats(script: str, duration_seconds: int) -> dict[str, Any]:
    effective = len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", script))
    estimated_seconds = round(effective / 4.2, 1) if effective else 0.0
    target_min = int(duration_seconds * 3.5)
    target_max = int(duration_seconds * 4.8)
    warnings: list[str] = []
    if effective < target_min:
        warnings.append(f"当前约 {effective} 个有效字，可能短于 {duration_seconds} 秒目标。")
    if effective > target_max:
        warnings.append(f"当前约 {effective} 个有效字，可能长于 {duration_seconds} 秒目标。")
    return {
        "effective_chars": effective,
        "estimated_seconds": estimated_seconds,
        "target_chars": [target_min, target_max],
        "warnings": warnings,
    }


def new_draft_id() -> str:
    return "script-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def create_draft() -> dict[str, Any]:
    draft_id = new_draft_id()
    now = _now()
    paths = get_studio_paths(ROOT)
    defaults = (
        SettingsStore(paths.appdata_root).load().get("defaults", {})
        if not paths.legacy_mode
        else {}
    )
    data = {
        "schema_version": 1,
        "draft_id": draft_id,
        "status": "editing",
        "mode": "direct",
        "title": "",
        "duration_seconds": None,
        "voice_profile": str(defaults.get("voice") or default_voice_profile()),
        "llm_profile": str(defaults.get("llm") or "deepseek_default"),
        "script_llm_profile": str(defaults.get("script_llm") or defaults.get("llm") or "deepseek_default"),
        "visual_llm_profile": str(defaults.get("visual_llm") or defaults.get("llm") or "deepseek_default"),
        "semantic_llm_profile": str(defaults.get("semantic_llm") or defaults.get("llm") or "deepseek_default"),
        "image_profile": str(defaults.get("image") or "comfyui_default"),
        "comfyui_workflow_profile": str(defaults.get("comfyui_workflow") or "history_image_default"),
        "subtitle_profile": str(defaults.get("subtitle") or "social_pink"),
        "candidates_per_shot": int(defaults.get("candidates_per_shot") or 4),
        "subtitle_overrides": {},
        "create_jianying_draft": True,
        "ai_disclosure": True,
        "visual_strategy": str(defaults.get("visual_strategy") or "museum_and_ai"),
        "topic": "",
        "source_material": "",
        "must_include": "",
        "avoid": "",
        "original_script": "",
        "final_script": "",
        "analysis": {"issues": [], "risks": [], "summary": ""},
        "suggestions": {"emphasis": [], "proper_nouns": [], "pronunciation": {}},
        "versions": [],
        "manual_versions": [],
        "ai_error": None,
        "created_at": now,
        "updated_at": now,
        "locked_project_id": None,
    }
    save_draft(data)
    return data


def create_revision_draft(parent_task_id: str) -> dict[str, Any]:
    """Create an editable draft derived from one successful sourced task."""
    paths = get_studio_paths(ROOT)
    run_dir = resolve_parent_task(paths.output_root, parent_task_id)
    task = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))
    parent_project_id = str(task.get("project_id") or "")
    parent_project = paths.project_root / parent_project_id
    if not parent_project.is_dir():
        legacy = ROOT / "projects" / parent_project_id
        parent_project = legacy if legacy.is_dir() else parent_project
    required = ["script.txt", "project.yaml", "profile_snapshot.json", "script_manifest.json"]
    missing = [name for name in required if not (parent_project / name).is_file()]
    if missing:
        raise ScriptWorkbenchError("父项目不完整，缺少：" + "、".join(missing))
    try:
        config = yaml.safe_load((parent_project / "project.yaml").read_text(encoding="utf-8"))
        snapshot = json.loads((parent_project / "profile_snapshot.json").read_text(encoding="utf-8"))
        manifest = json.loads((parent_project / "script_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ScriptWorkbenchError("父项目配置无法读取。") from exc
    if not isinstance(config, dict) or not isinstance(snapshot, dict) or not isinstance(manifest, dict):
        raise ScriptWorkbenchError("父项目配置格式无效。")
    data = create_draft()
    profiles = snapshot.get("profiles") if isinstance(snapshot.get("profiles"), dict) else {}
    profile_id = lambda name, fallback: str((profiles.get(name) or {}).get("id") or fallback)
    overrides = snapshot.get("project_overrides") if isinstance(snapshot.get("project_overrides"), dict) else {}
    script = (parent_project / "script.txt").read_text(encoding="utf-8")
    data.update(
        {
            "mode": "direct",
            "title": str(config.get("project", {}).get("title") or parent_project_id) + "（修订）",
            "duration_seconds": int(manifest.get("target_duration_seconds") or 60),
            "voice_profile": profile_id("voice", manifest.get("voice_profile") or default_voice_profile()),
            "llm_profile": profile_id("default_llm", "deepseek_default"),
            "script_llm_profile": profile_id("script_llm", "deepseek_default"),
            "visual_llm_profile": profile_id("visual_llm", "deepseek_default"),
            "semantic_llm_profile": profile_id("semantic_llm", "deepseek_default"),
            "image_profile": profile_id("image", "comfyui_default"),
            "comfyui_workflow_profile": profile_id("comfyui_workflow", "history_image_default"),
            "subtitle_profile": profile_id("subtitle", "social_pink"),
            "candidates_per_shot": int(overrides.get("candidates_per_shot") or config.get("visuals", {}).get("ai_fallback", {}).get("candidates_per_shot", 4)),
            "subtitle_overrides": dict(overrides.get("subtitle") or {}),
            "create_jianying_draft": bool(overrides.get("create_jianying_draft", True)),
            "ai_disclosure": bool(overrides.get("ai_disclosure", True)),
            "visual_strategy": str(overrides.get("visual_strategy") or config.get("visuals", {}).get("strategy") or "museum_and_ai"),
            "original_script": script,
            "final_script": script,
            "revision": {
                "parent_project_id": parent_project_id,
                "parent_task_id": str(task.get("task_id") or run_dir.name),
                "parent_script_sha256": _hash_text(script),
                "scope": "selected_and_ai",
                "generate_for_reused": False,
                "max_uses_per_asset": 1,
                "allow_manual_duplicate": True,
                "recommendation_threshold": 75,
                "alternative_threshold": 55,
                "require_review": True,
            },
        }
    )
    save_draft(data)
    return data


def draft_path(draft_id: str) -> Path:
    if not re.fullmatch(r"script-[0-9_]+", draft_id):
        raise ScriptWorkbenchError("脚本草稿 ID 无效。")
    return DRAFTS_ROOT / draft_id / "draft.json"


def save_draft(data: dict[str, Any]) -> None:
    data["updated_at"] = _now()
    _atomic_json(draft_path(str(data["draft_id"])), data)


def load_draft(draft_id: str) -> dict[str, Any]:
    path = draft_path(draft_id)
    if not path.is_file():
        raise ScriptWorkbenchError("找不到脚本草稿。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ScriptWorkbenchError("脚本草稿版本不受支持。")
    return payload


def latest_editing_draft() -> dict[str, Any] | None:
    candidates: list[tuple[float, Path]] = []
    if not DRAFTS_ROOT.is_dir():
        return None
    for path in DRAFTS_ROOT.glob("script-*/draft.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "editing":
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        return None
    return json.loads(max(candidates, key=lambda item: item[0])[1].read_text(encoding="utf-8"))


def discard_draft(data: dict[str, Any]) -> None:
    data["status"] = "discarded"
    save_draft(data)


def _clean_string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text[:80])
        if len(result) >= limit:
            break
    return result


def _validate_ai_result(result: dict[str, Any], mode: str) -> dict[str, Any]:
    suggested_script = _normalize_script(str(result.get("suggested_script") or "")).strip()
    if not suggested_script:
        raise ScriptWorkbenchError("DeepSeek 没有返回可用脚本。")
    issues = []
    for item in result.get("issues", []) if isinstance(result.get("issues"), list) else []:
        if isinstance(item, dict):
            issues.append(
                {
                    "category": str(item.get("category") or "内容").strip()[:30],
                    "message": str(item.get("message") or "").strip()[:500],
                    "suggestion": str(item.get("suggestion") or "").strip()[:500],
                }
            )
    risks = []
    for item in result.get("risks", []) if isinstance(result.get("risks"), list) else []:
        if isinstance(item, dict):
            risks.append(
                {
                    "type": str(item.get("type") or "uncertain_fact").strip()[:40],
                    "message": str(item.get("message") or "").strip()[:500],
                }
            )
    suggestions = result.get("suggestions") if isinstance(result.get("suggestions"), dict) else {}
    pronunciation = suggestions.get("pronunciation") if isinstance(suggestions.get("pronunciation"), dict) else {}
    clean_pronunciation = {
        str(key).strip()[:40]: str(value).strip()[:80]
        for key, value in list(pronunciation.items())[:12]
        if str(key).strip() and str(value).strip()
    }
    return {
        "suggested_script": suggested_script,
        "analysis": {
            "issues": issues[:20],
            "risks": risks[:20],
            "summary": str(result.get("summary") or "").strip()[:1000],
        },
        "suggestions": {
            "emphasis": _clean_string_list(suggestions.get("emphasis")),
            "proper_nouns": _clean_string_list(suggestions.get("proper_nouns")),
            "pronunciation": clean_pronunciation,
        },
        "mode": mode,
    }


def _ai_settings(data: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    profile_id = str((data or {}).get("script_llm_profile") or "deepseek_default")
    try:
        profile = ProfileStore(get_studio_paths(ROOT)).get("llm", profile_id)
    except ProfileError as exc:
        raise ScriptWorkbenchError(str(exc)) from exc
    if profile.get("protocol") != "chat_completions":
        raise ScriptWorkbenchError(
            "脚本工作台当前要求 Chat Completions 兼容配置档；OpenAI Responses 可用于镜头规划。"
        )
    env = load_local_env(ROOT)
    secret_ref = str(profile.get("secret_ref") or "DEEPSEEK_API_KEY")
    return {
        "base_url": profile.get("base_url"),
        "model": profile.get("model"),
        "thinking": profile.get("thinking", "disabled"),
        "max_tokens": int(profile.get("max_tokens", 8000)),
        "timeout_seconds": float(profile.get("timeout_seconds", 120)),
    }, env.get(secret_ref, "")


def _prompt() -> str:
    return """你是中文短视频历史趣闻脚本编辑。只输出 JSON 对象，不要输出 Markdown。
文风要求：开头直接给反常识钩子；短句、口语化、逐层推进；不要片头、自我介绍、点赞关注；结尾落在反转或余味。禁止编造引语。
你只做内容质量判断，不是史实来源。无法确认的具体人名、数字、年代、因果和引语必须放入 risks，不得写成已经联网核实。
JSON 必须包含：suggested_script 字符串；summary 字符串；issues 数组，每项包含 category/message/suggestion；risks 数组，每项包含 type/message；suggestions 对象，含 emphasis 数组、proper_nouns 数组、pronunciation 对象。
review 模式必须保留原稿核心事实和叙事意图；topic 模式按用户资料写初稿，不得补充无法确认的精确引语。"""


def _cache_path(payload: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return get_studio_paths(ROOT).cache_root / "script_workbench" / f"{key}.json"


def save_manual_version(data: dict[str, Any], summary: str = "手动保存") -> dict[str, Any] | None:
    script = str(data.get("final_script") or data.get("original_script") or "")
    if not script.strip():
        return None
    manual_versions = data.setdefault("manual_versions", [])
    if manual_versions and manual_versions[-1].get("script") == script:
        return manual_versions[-1]
    version = {
        "version": len(manual_versions) + 1,
        "created_at": _now(),
        "summary": summary.strip()[:300] or "手动保存",
        "script": script,
        "based_on_ai_version": len(data.get("versions", [])) or None,
    }
    manual_versions.append(version)
    save_draft(data)
    return version


def run_ai_revision(
    data: dict[str, Any], feedback: str = "", *, requester: Callable[..., Any] = request_json_object
) -> dict[str, Any]:
    mode = str(data.get("mode"))
    if mode not in {"review", "topic"}:
        raise ScriptWorkbenchError("direct 模式不会调用 DeepSeek。")
    if len(data.get("versions", [])) >= MAX_AI_VERSIONS:
        raise ScriptWorkbenchError("该草稿已达到 10 次 AI 版本上限，请直接人工编辑。")
    if data.get("final_script"):
        latest_ai = (data.get("versions") or [{}])[-1].get("script")
        if data.get("final_script") != latest_ai:
            save_manual_version(data, "生成新 AI 版本前的人工稿")
    duration = int(data.get("duration_seconds") or 0)
    payload = {
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "title": str(data.get("title", "")),
        "target_duration_seconds": duration,
        "original_script": str(data.get("original_script", "")),
        "current_script": str(data.get("final_script", "")),
        "topic": str(data.get("topic", "")),
        "source_material": str(data.get("source_material", "")),
        "must_include": str(data.get("must_include", "")),
        "avoid": str(data.get("avoid", "")),
        "revision_feedback": feedback.strip(),
    }
    cache_path = _cache_path(payload)
    safe: dict[str, Any]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        result = _validate_ai_result(cached["result"], mode)
        safe = {**cached.get("response", {}), "cache_hit": True}
    else:
        settings, api_key = _ai_settings(data)
        if not api_key:
            raise ScriptWorkbenchError("缺少 DEEPSEEK_API_KEY；草稿已保留，可稍后重试或切换 direct。")
        try:
            raw, safe = requester(_prompt(), payload, settings, api_key)
        except DeepSeekPlannerError as exc:
            raise ScriptWorkbenchError(str(exc)) from exc
        result = _validate_ai_result(raw, mode)
        _atomic_json(cache_path, {"result": raw, "response": safe})
    if mode == "topic" and not str(data.get("source_material", "")).strip() and not result["analysis"]["risks"]:
        result["analysis"]["risks"].append(
            {
                "type": "source_missing",
                "message": "本稿仅根据选题和模型知识生成，具体事实尚未提供资料或联网核验。",
            }
        )
    version = {
        "version": len(data.get("versions", [])) + 1,
        "mode": mode,
        "created_at": _now(),
        "feedback": feedback.strip(),
        "script": result["suggested_script"],
        "analysis": result["analysis"],
        "suggestions": result["suggestions"],
        "provider": "deepseek",
        "model": safe.get("model", "deepseek-v4-flash"),
        "prompt_version": PROMPT_VERSION,
        "cache_hit": bool(safe.get("cache_hit", False)),
    }
    data.setdefault("versions", []).append(version)
    data["final_script"] = version["script"]
    data["analysis"] = version["analysis"]
    data["suggestions"] = version["suggestions"]
    data["ai_error"] = None
    save_draft(data)
    return version


def update_from_form(data: dict[str, Any], values: dict[str, list[str]]) -> None:
    def value(name: str) -> str:
        if name in values:
            return values[name][0]
        current = data.get(name, "")
        return "" if current is None else str(current)

    mode = value("mode") or str(data.get("mode", "direct"))
    if mode not in ALLOWED_MODES:
        raise ScriptWorkbenchError("脚本模式无效。")
    duration_text = value("duration_seconds")
    duration = int(duration_text) if duration_text.isdigit() else data.get("duration_seconds")
    if duration is not None and int(duration) not in ALLOWED_DURATIONS:
        raise ScriptWorkbenchError("目标时长只能选择 30、45、60 或 90 秒。")
    visual_strategy = value("visual_strategy") or str(
        data.get("visual_strategy", "museum_and_ai")
    )
    if visual_strategy not in ALLOWED_VISUAL_STRATEGIES:
        raise ScriptWorkbenchError("画面来源只能选择馆藏＋AI、纯 AI 或本地素材。")
    voice_profiles = load_voice_profiles()
    voice_profile = value("voice_profile") or str(
        data.get("voice_profile") or default_voice_profile()
    )
    if voice_profile not in voice_profiles:
        raise ScriptWorkbenchError("选择的配音 profile 不存在或已被移除。")
    profile_store = ProfileStore(get_studio_paths(ROOT))
    profile_fields = {
        "llm_profile": ("llm", "deepseek_default"),
        "script_llm_profile": ("llm", "deepseek_default"),
        "visual_llm_profile": ("llm", "deepseek_default"),
        "semantic_llm_profile": ("llm", "deepseek_default"),
        "image_profile": ("image", "comfyui_default"),
        "comfyui_workflow_profile": ("comfyui_workflow", "history_image_default"),
        "subtitle_profile": ("subtitle", "social_pink"),
    }
    selected_profiles: dict[str, str] = {}
    for field, (kind, fallback) in profile_fields.items():
        selected = value(field) or str(data.get(field) or fallback)
        try:
            profile_store.get(kind, selected)
        except ProfileError as exc:
            raise ScriptWorkbenchError(str(exc)) from exc
        selected_profiles[field] = selected
    candidates_text = value("candidates_per_shot") or str(data.get("candidates_per_shot", 4))
    if not candidates_text.isdigit() or not 1 <= int(candidates_text) <= 8:
        raise ScriptWorkbenchError("每个镜头的 AI 候选数量必须在 1–8 之间。")
    subtitle_overrides = dict(data.get("subtitle_overrides") or {})
    subtitle_fields: dict[str, tuple[str, Callable[[str], Any], tuple[float, float]]] = {
        "font_name": ("subtitle_font_name", str, (0, 0)),
        "font_size": ("subtitle_font_size", int, (36, 180)),
        "base_color": ("subtitle_base_color", str, (0, 0)),
        "highlight_color": ("subtitle_highlight_color", str, (0, 0)),
        "outline": ("subtitle_outline", int, (0, 20)),
        "shadow": ("subtitle_shadow", int, (0, 20)),
        "margin_bottom": ("subtitle_margin_bottom", int, (0, 1000)),
        "max_chars_per_line": ("subtitle_max_chars", int, (1, 24)),
        "max_lines": ("subtitle_max_lines", int, (1, 2)),
        "fade_in_ms": ("subtitle_fade_in_ms", int, (0, 2000)),
        "fade_out_ms": ("subtitle_fade_out_ms", int, (0, 2000)),
    }
    for output_key, (form_key, converter, bounds) in subtitle_fields.items():
        raw = value(form_key).strip()
        if not raw:
            continue
        try:
            converted = converter(raw)
        except ValueError as exc:
            raise ScriptWorkbenchError(f"字幕参数 {form_key} 格式无效。") from exc
        if converter is int and not bounds[0] <= converted <= bounds[1]:
            raise ScriptWorkbenchError(f"字幕参数 {form_key} 超出允许范围。")
        if output_key.endswith("color") and not re.fullmatch(r"#[0-9A-Fa-f]{6}", str(converted)):
            raise ScriptWorkbenchError(f"字幕颜色 {form_key} 必须是 #RRGGBB。")
        subtitle_overrides[output_key] = converted
    data.update(
        {
            "mode": mode,
            "title": value("title").strip()[:160],
            "duration_seconds": int(duration) if duration is not None else None,
            "voice_profile": voice_profile,
            "visual_strategy": visual_strategy,
            **selected_profiles,
            "candidates_per_shot": int(candidates_text),
            "subtitle_overrides": subtitle_overrides,
            "create_jianying_draft": value("create_jianying_draft") == "1"
            if "create_jianying_draft_present" in values
            else bool(data.get("create_jianying_draft", True)),
            "ai_disclosure": value("ai_disclosure") == "1"
            if "ai_disclosure_present" in values
            else bool(data.get("ai_disclosure", True)),
            "topic": value("topic").strip()[:4000],
            "source_material": value("source_material").strip()[:20000],
            "must_include": value("must_include").strip()[:4000],
            "avoid": value("avoid").strip()[:4000],
            "original_script": _normalize_script(value("original_script"))[:30000],
            "final_script": _normalize_script(value("final_script"))[:30000],
        }
    )
    if any(name in values for name in ("emphasis", "proper_nouns", "pronunciation")):
        suggestions = data.setdefault("suggestions", {})
        suggestions["emphasis"] = _parse_lines(value("emphasis"))
        suggestions["proper_nouns"] = _parse_lines(value("proper_nouns"))
        pronunciation: dict[str, str] = {}
        for line in value("pronunciation").splitlines():
            if "=" not in line:
                continue
            key, reading = line.split("=", 1)
            if key.strip() and reading.strip():
                pronunciation[key.strip()[:40]] = reading.strip()[:80]
        suggestions["pronunciation"] = pronunciation
    save_draft(data)


def _parse_lines(value: str, *, limit: int = 12) -> list[str]:
    items = re.split(r"[,，;；\n]+", value)
    return [item.strip() for item in items if item.strip()][:limit]


def _project_id() -> str:
    base = "history-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base
    index = 2
    while (get_studio_paths(ROOT).project_root / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def lock_draft(data: dict[str, Any], values: dict[str, list[str]]) -> Path:
    update_from_form(data, values)
    title = str(data.get("title", "")).strip()
    duration = data.get("duration_seconds")
    final_candidate = str(data.get("final_script") or "")
    script = _normalize_script(
        final_candidate if final_candidate.strip() else str(data.get("original_script") or "")
    )
    if not title:
        raise ScriptWorkbenchError("锁稿前必须填写视频标题。")
    if duration not in ALLOWED_DURATIONS:
        raise ScriptWorkbenchError("锁稿前必须选择目标时长。")
    if data.get("mode") == "topic" and not any(
        item.get("mode") == "topic" for item in data.get("versions", [])
    ):
        raise ScriptWorkbenchError("topic 模式尚未成功生成初稿，不能创建空项目。")
    if not script.strip():
        raise ScriptWorkbenchError("锁稿前必须提供最终脚本。")
    project_id = _project_id()
    paths = get_studio_paths(ROOT)
    paths.project_root.mkdir(parents=True, exist_ok=True)
    final_dir = paths.project_root / project_id
    temporary = paths.project_root / f".{project_id}.tmp-{secrets.token_hex(4)}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        revision = data.get("revision") if isinstance(data.get("revision"), dict) else None
        if revision:
            parent_project_id = str(revision.get("parent_project_id") or "")
            parent_project = paths.project_root / parent_project_id
            if not parent_project.is_dir() and (ROOT / "projects" / parent_project_id).is_dir():
                parent_project = ROOT / "projects" / parent_project_id
            parent_config_path = parent_project / "project.yaml"
            if not parent_config_path.is_file():
                raise ScriptWorkbenchError("父项目配置已丢失，不能建立派生项目。")
            config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
        else:
            config = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
        config["project"]["id"] = project_id
        config["project"]["title"] = title
        profile_store = ProfileStore(paths)
        voice_profile = profile_store.get("voice", str(data.get("voice_profile") or default_voice_profile()))
        subtitle_profile = profile_store.get("subtitle", str(data.get("subtitle_profile") or "social_pink"))
        image_profile = profile_store.get("image", str(data.get("image_profile") or "comfyui_default"))
        workflow_profile = profile_store.get(
            "comfyui_workflow",
            str(data.get("comfyui_workflow_profile") or "history_image_default"),
        )
        visual_llm = profile_store.get("llm", str(data.get("visual_llm_profile") or "deepseek_default"))
        semantic_llm = profile_store.get("llm", str(data.get("semantic_llm_profile") or visual_llm["id"]))
        config["voice"] = {
            "provider": "moneyprinter_edge",
            "profile": voice_profile["id"],
            "voice": voice_profile["voice"],
            "label": voice_profile["name"],
            "rate": voice_profile.get("rate", "+0%"),
            "pitch": voice_profile.get("pitch", "+0Hz"),
            "pronunciation": {},
            "pauses": {},
        }
        config["subtitles"]["style_preset"] = str(subtitle_profile.get("preset") or subtitle_profile["id"])
        config["subtitles"].update(data.get("subtitle_overrides") or {})
        config["visuals"]["strategy"] = str(
            data.get("visual_strategy", "museum_and_ai")
        )
        protocol = str(visual_llm.get("protocol"))
        config["visuals"]["planner"].update(
            {
                "provider": "openai" if protocol == "openai_responses" else "chat_completions",
                "base_url": visual_llm.get("base_url"),
                "model": visual_llm.get("model"),
                "timeout_seconds": visual_llm.get("timeout_seconds", 120),
                "max_tokens": visual_llm.get("max_tokens", 12000),
                "secret_ref": visual_llm.get("secret_ref"),
            }
        )
        config["visuals"]["planner"]["semantic_audit"].update(
            {
                "provider": "chat_completions",
                "base_url": semantic_llm.get("base_url"),
                "model": semantic_llm.get("model"),
                "secret_ref": semantic_llm.get("secret_ref"),
            }
        )
        config["visuals"]["search"]["semantic_review"].update(
            {
                "provider": "chat_completions",
                "base_url": semantic_llm.get("base_url"),
                "model": semantic_llm.get("model"),
                "secret_ref": semantic_llm.get("secret_ref"),
            }
        )
        ai_config = config["visuals"]["ai_fallback"]
        if image_profile.get("protocol") == "comfyui_local":
            source_workflow = Path(str(workflow_profile.get("workflow_file") or "history_image_api.json"))
            if not source_workflow.is_absolute():
                source_workflow = (
                    paths.workflow_root / source_workflow
                    if (paths.workflow_root / source_workflow).is_file()
                    else ROOT / source_workflow
                    if (ROOT / source_workflow).is_file()
                    else Path(__file__).resolve().parents[1] / source_workflow
                )
            workflow_copy = temporary / "workflow.json"
            shutil.copy2(source_workflow, workflow_copy)
            ai_config.update(
                {
                    "provider": "comfyui_local",
                    "server_url": image_profile.get("server_url", "http://127.0.0.1:8000"),
                    "workflow_file": "workflow.json",
                    "prompt_marker": workflow_profile.get("prompt_marker", "__AI_VIDEO_PROMPT__"),
                    "bindings": workflow_profile.get("bindings", {}),
                    "width": int(workflow_profile.get("width", 768)),
                    "height": int(workflow_profile.get("height", 1344)),
                    "max_images_per_run": "all_candidates",
                }
            )
        else:
            ai_config.update(
                {
                    "provider": "openai",
                    "base_url": image_profile.get("base_url"),
                    "model": image_profile.get("model"),
                    "size": image_profile.get("size", "1024x1536"),
                    "quality": image_profile.get("quality", "medium"),
                    "output_format": image_profile.get("output_format", "jpeg"),
                    "secret_ref": image_profile.get("secret_ref"),
                    "max_images_per_run": int(image_profile.get("max_images_per_run", 4)),
                }
            )
        ai_config["candidates_per_shot"] = int(data.get("candidates_per_shot", 4))
        config["visuals"]["review"]["disclosure"] = "end_only" if data.get("ai_disclosure", True) else "disabled"
        config["jianying"]["draft_root"] = str(
            Path(
                str(
                    SettingsStore(paths.appdata_root).load().get("jianying_draft_root")
                    or discover_jianying_draft_root()
                )
            ).resolve()
        )
        config["media"]["seed"] = int(hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:8], 16)
        config["jianying"]["draft_name_prefix"] = f"AI-History-{project_id.removeprefix('history-')}"
        if revision:
            try:
                reuse_snapshot = create_reuse_source_snapshot(
                    paths.output_root,
                    paths.cache_root,
                    str(revision.get("parent_task_id") or ""),
                )
            except AssetReuseError as exc:
                raise ScriptWorkbenchError(str(exc)) from exc
            _atomic_json(temporary / "reuse_source_snapshot.json", reuse_snapshot)
            config["revision"] = {
                "parent_project_id": reuse_snapshot["parent_project_id"],
                "parent_task_id": reuse_snapshot["parent_task_id"],
                "reuse_snapshot_file": "reuse_source_snapshot.json",
            }
            config.setdefault("visuals", {})["reuse"] = {
                "enabled": True,
                "scope": "selected_and_ai",
                "generate_for_reused": False,
                "max_uses_per_asset": int(revision.get("max_uses_per_asset", 1)),
                "allow_manual_duplicate": bool(revision.get("allow_manual_duplicate", True)),
                "recommendation_threshold": int(revision.get("recommendation_threshold", 75)),
                "alternative_threshold": int(revision.get("alternative_threshold", 55)),
                "shots_per_batch": 4,
                "require_review": bool(revision.get("require_review", True)),
            }
        suggestions = data.get("suggestions") if isinstance(data.get("suggestions"), dict) else {}
        emphasis = config["subtitles"]["emphasis"]
        emphasis["include"] = _parse_lines(values.get("emphasis", [""])[0]) or _clean_string_list(suggestions.get("emphasis"))
        emphasis["proper_nouns"] = _parse_lines(values.get("proper_nouns", [""])[0]) or _clean_string_list(suggestions.get("proper_nouns"))
        pronunciation_text = values.get("pronunciation", [""])[0]
        pronunciation: dict[str, str] = {}
        for line in pronunciation_text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip() and value.strip():
                    pronunciation[key.strip()] = value.strip()
        if not pronunciation and isinstance(suggestions.get("pronunciation"), dict):
            pronunciation = {str(k): str(v) for k, v in suggestions["pronunciation"].items()}
        config["voice"]["pronunciation"] = pronunciation
        selections = {
            "default_llm": f"llm:{data.get('llm_profile', 'deepseek_default')}",
            "script_llm": f"llm:{data.get('script_llm_profile', 'deepseek_default')}",
            "visual_llm": f"llm:{data.get('visual_llm_profile', 'deepseek_default')}",
            "semantic_llm": f"llm:{data.get('semantic_llm_profile', 'deepseek_default')}",
            "image": f"image:{data.get('image_profile', 'comfyui_default')}",
            "comfyui_workflow": f"comfyui_workflow:{data.get('comfyui_workflow_profile', 'history_image_default')}",
            "voice": f"voice:{data.get('voice_profile', 'yunyang_soft')}",
            "subtitle": f"subtitle:{data.get('subtitle_profile', 'social_pink')}",
        }
        snapshot = profile_store.snapshot(selections)
        snapshot["project_overrides"] = {
            "visual_strategy": data.get("visual_strategy"),
            "candidates_per_shot": data.get("candidates_per_shot"),
            "subtitle": data.get("subtitle_overrides"),
            "create_jianying_draft": data.get("create_jianying_draft"),
            "ai_disclosure": data.get("ai_disclosure"),
        }
        if revision:
            snapshot["derived_from"] = {
                "parent_project_id": revision.get("parent_project_id"),
                "parent_task_id": revision.get("parent_task_id"),
                "parent_script_sha256": revision.get("parent_script_sha256"),
                "reuse_snapshot_sha256": reuse_snapshot.get("sha256"),
                "derived_at": _now(),
            }
        snapshot["sha256"] = snapshot_hash(snapshot)
        _atomic_json(temporary / "profile_snapshot.json", snapshot)
        (temporary / "script.txt").write_text(script, encoding="utf-8")
        (temporary / "project.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=1000),
            encoding="utf-8",
        )
        history_dir = temporary / "script_history"
        history_dir.mkdir()
        (history_dir / "original.txt").write_text(str(data.get("original_script", "")), encoding="utf-8")
        for version in data.get("versions", []):
            number = int(version.get("version", 0))
            _atomic_json(history_dir / f"version-{number:02d}.json", version)
        for version in data.get("manual_versions", []):
            number = int(version.get("version", 0))
            _atomic_json(history_dir / f"manual-{number:02d}.json", version)
        (history_dir / "final.txt").write_text(script, encoding="utf-8")
        _atomic_json(
            history_dir / "final-edit.json",
            {
                "based_on_ai_version": len(data.get("versions", [])) or None,
                "final_script_sha256": _hash_text(script),
                "differs_from_latest_ai_version": bool(data.get("versions"))
                and script != str(data["versions"][-1].get("script", "")),
                "saved_at": _now(),
            },
        )
        manifest = {
            "schema_version": 1,
            "project_id": project_id,
            "mode": data.get("mode"),
            "target_duration_seconds": duration,
            "voice_profile": data.get("voice_profile") or default_voice_profile(),
            "visual_strategy": data.get("visual_strategy", "museum_and_ai"),
            "original_script_sha256": _hash_text(str(data.get("original_script", ""))),
            "final_script_sha256": _hash_text(script),
            "effective_chars": script_stats(script, int(duration))["effective_chars"],
            "ai_versions": len(data.get("versions", [])),
            "manual_versions": len(data.get("manual_versions", [])),
            "risks": data.get("analysis", {}).get("risks", []),
            "provider": "deepseek" if data.get("versions") else None,
            "model": (data.get("versions") or [{}])[-1].get("model"),
            "prompt_version": PROMPT_VERSION if data.get("versions") else None,
            "locked_at": _now(),
            "publish_ready": False,
        }
        if revision:
            manifest["derived_from"] = {
                "parent_project_id": revision.get("parent_project_id"),
                "parent_task_id": revision.get("parent_task_id"),
                "parent_script_sha256": revision.get("parent_script_sha256"),
                "reuse_snapshot_sha256": reuse_snapshot.get("sha256"),
                "derived_at": _now(),
            }
        _atomic_json(temporary / "script_manifest.json", manifest)
        os.replace(temporary, final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    data["status"] = "locked"
    data["locked_project_id"] = project_id
    data["final_script"] = script
    save_draft(data)
    return final_dir


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _latest_video_task() -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    outputs = get_studio_paths(ROOT).output_root
    if not outputs.is_dir():
        return None
    for path in outputs.glob("*/task.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project_id = str(data.get("project_id") or "")
        paths = get_studio_paths(ROOT)
        if not project_id or not (
            (paths.project_root / project_id).is_dir()
            or (ROOT / "projects" / project_id).is_dir()
        ):
            continue
        if data.get("status") in {"failed", "waiting_for_review"}:
            data["run_dir"] = str(path.parent)
            candidates.append((path.stat().st_mtime, data))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _form(data: dict[str, Any], csrf: str, message: str = "") -> str:
    mode = str(data.get("mode", "direct"))
    duration = data.get("duration_seconds")
    voice_profiles = load_voice_profiles()
    voice_profile = str(data.get("voice_profile") or default_voice_profile())
    if voice_profile not in voice_profiles:
        voice_profile = default_voice_profile()
    previews = voice_preview_files()
    voice_options = "".join(
        f'<option value="{_escape(profile_id)}" '
        f'data-preview="{f"/voice-preview/{profile_id}" if profile_id in previews else ""}" '
        f'{"selected" if profile_id == voice_profile else ""}>'
        f'{_escape(profile["label"])} · {_escape(profile["voice"])} · '
        f'{_escape(profile["rate"])} / {_escape(profile["pitch"])}</option>'
        for profile_id, profile in voice_profiles.items()
    )
    preview_url = f"/voice-preview/{voice_profile}" if voice_profile in previews else ""
    visual_strategy = str(data.get("visual_strategy", "museum_and_ai"))
    versions = list(data.get("versions", []))
    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
    suggestions = data.get("suggestions") if isinstance(data.get("suggestions"), dict) else {}
    stats = script_stats(str(data.get("final_script") or data.get("original_script") or ""), int(duration or 45))
    issues = "".join(
        f"<li><b>{_escape(item.get('category'))}</b>：{_escape(item.get('message'))}<small>{_escape(item.get('suggestion'))}</small></li>"
        for item in analysis.get("issues", [])
    ) or "<li>尚无 AI 审稿结果。</li>"
    risks = "".join(f"<li>{_escape(item.get('message'))}</li>" for item in analysis.get("risks", [])) or "<li>尚无风险提示；这不代表已经完成史实核查。</li>"
    version_options = "".join(
        f'<option value="ai:{int(item.get("version", 0))}">AI {int(item.get("version", 0))} · {_escape(item.get("feedback") or "初稿")}</option>'
        for item in versions
    )
    version_options += "".join(
        f'<option value="manual:{int(item.get("version", 0))}">人工 {int(item.get("version", 0))} · {_escape(item.get("summary") or "手动保存")}</option>'
        for item in data.get("manual_versions", [])
    )
    pronunciation = "\n".join(f"{key}={value}" for key, value in (suggestions.get("pronunciation") or {}).items())
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Video 脚本工作台</title>
<style>:root{{--bg:#0e0f12;--panel:#191b20;--panel2:#22252c;--text:#f4f2ed;--muted:#a9abb1;--gold:#ffd54a;--red:#ff9d95}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1380px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:20px}}h1{{margin:0}}.muted,small{{color:var(--muted)}}.error{{background:#4a2020;color:#ffd2cf;padding:10px 14px;border-radius:9px;margin:12px 0}}.top,.columns,.meta-grid{{display:grid;gap:15px}}.top{{grid-template-columns:1fr 1fr 1fr}}.columns{{grid-template-columns:1fr 1fr}}.meta-grid{{grid-template-columns:1fr 1fr 1fr}}label{{display:flex;flex-direction:column;gap:6px;color:#dedbd2}}input,select,textarea{{width:100%;background:var(--panel);border:1px solid #383c45;border-radius:9px;color:white;padding:11px;font:inherit}}textarea{{resize:vertical;min-height:120px}}textarea.script{{min-height:430px;font-size:17px;line-height:1.75}}section{{background:var(--panel);border:1px solid #2d3037;border-radius:13px;padding:17px;margin:15px 0}}button{{border:0;border-radius:9px;background:var(--gold);color:#171717;padding:11px 17px;font-weight:750;cursor:pointer}}button.secondary{{background:#343840;color:#eee}}button.danger{{background:#53272a;color:#ffd8d8}}.actions{{display:flex;gap:10px;align-items:end;flex-wrap:wrap}}ul{{padding-left:21px}}li small{{display:block;margin:3px 0 9px}}.risk{{border-color:#6b4b2d}}.risk li{{color:#ffd3a7}}details{{margin-top:10px}}@media(max-width:900px){{.top,.columns,.meta-grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><div><h1>脚本工作台</h1><div class="muted">锁稿前不会生成配音、搜索素材或调用 ComfyUI。</div></div><form method="post" action="/action"><input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="action" value="discard"><button class="danger">放弃草稿并新建</button></form></header>
{f'<div class="error">{_escape(message)}</div>' if message else ''}<form id="script-form" method="post" action="/action"><input type="hidden" name="csrf" value="{csrf}"><input type="hidden" id="action" name="action" value="save">
<section class="top"><label>模式<select name="mode"><option value="direct" {'selected' if mode=='direct' else ''}>direct · 我提供成稿</option><option value="review" {'selected' if mode=='review' else ''}>review · AI 审稿</option><option value="topic" {'selected' if mode=='topic' else ''}>topic · AI 生成初稿</option></select></label><label>视频标题<input name="title" maxlength="160" value="{_escape(data.get('title'))}" required></label><label>目标时长<select name="duration_seconds" required><option value="">请选择</option>{''.join(f'<option value="{value}" {"selected" if duration==value else ""}>{value} 秒</option>' for value in (30,45,60,90))}</select></label></section>
<section><div class="columns"><label>配音<select id="voice-profile" name="voice_profile">{voice_options}</select><audio id="voice-preview" controls preload="none" {'src='+preview_url if preview_url else ''}></audio><div id="voice-preview-note" class="muted">{'点击播放可直接试听同一段对比文案。' if preview_url else '该 profile 还没有本地试听文件，可运行 tools/voice_audition.bat 生成。'}</div></label><label>画面来源<select name="visual_strategy"><option value="museum_and_ai" {'selected' if visual_strategy=='museum_and_ai' else ''}>馆藏素材＋AI 候选</option><option value="ai_only" {'selected' if visual_strategy=='ai_only' else ''}>纯 AI · 跳过第 6 步网络素材搜索，每个镜头生成 4 张 ComfyUI 候选</option></select><div class="muted">纯 AI 仍会使用 DeepSeek 完成第 5 步镜头规划；只是不会请求馆藏站点。</div></label></div></section>
<section><div class="meta-grid"><label>选题（topic）<textarea name="topic">{_escape(data.get('topic'))}</textarea></label><label>补充资料/已知事实<textarea name="source_material">{_escape(data.get('source_material'))}</textarea></label><label>必须讲 / 不要讲<textarea name="must_include" placeholder="必须讲">{_escape(data.get('must_include'))}</textarea><textarea name="avoid" placeholder="不要讲">{_escape(data.get('avoid'))}</textarea></label></div></section>
<div class="columns"><section><h2>原稿</h2><textarea class="script" name="original_script" placeholder="direct/review 在这里粘贴你的完整脚本">{_escape(data.get('original_script'))}</textarea></section><section><h2>最终稿（可编辑）</h2><textarea class="script" name="final_script" placeholder="direct 可粘贴成稿；review/topic 会显示 AI 建议稿">{_escape(data.get('final_script'))}</textarea></section></div>
<section><b>约 {stats['effective_chars']} 个有效字 · 预计 {stats['estimated_seconds']} 秒 · 当前时长目标建议 {stats['target_chars'][0]}–{stats['target_chars'][1]} 字</b><div class="muted">{_escape('；'.join(stats['warnings']) or '长度处于估算范围内。实际时长以配音为准。')}</div></section>
<div class="columns"><section><h2>审稿问题</h2><p>{_escape(analysis.get('summary'))}</p><ul>{issues}</ul></section><section class="risk"><h2>风险提示</h2><div class="muted">仅提示，不阻止锁稿；AI 判断不等于史实核查。</div><ul>{risks}</ul></section></div>
<section><h2>AI 版本</h2><div class="actions"><select name="version_to_restore"><option value="">选择历史版本</option>{version_options}</select><button type="button" class="secondary" onclick="submitAction('restore')">恢复到编辑区</button><input name="feedback" placeholder="例如：更口语化，但保留最后一句"><button type="button" onclick="submitAction('ai')">按要求再改一版</button></div></section>
<details><summary>强调词、专名与读音（可选）</summary><section class="meta-grid"><label>强调词，逗号或换行分隔<textarea name="emphasis">{_escape('，'.join(suggestions.get('emphasis') or []))}</textarea></label><label>专名，逗号或换行分隔<textarea name="proper_nouns">{_escape('，'.join(suggestions.get('proper_nouns') or []))}</textarea></label><label>读音词典，每行 原词=读法<textarea name="pronunciation">{_escape(pronunciation)}</textarea></label></section></details>
<section class="actions"><button type="button" class="secondary" onclick="submitAction('save')">保存草稿</button><button type="button" onclick="submitAction('ai')">生成 / 审核脚本</button><button type="button" onclick="submitAction('lock')">锁定脚本并开始制作</button></section></form>
<script>function submitAction(value){{document.getElementById('action').value=value;document.getElementById('script-form').submit()}}const voiceSelect=document.getElementById('voice-profile'),voiceAudio=document.getElementById('voice-preview'),voiceNote=document.getElementById('voice-preview-note');voiceSelect.addEventListener('change',()=>{{const url=voiceSelect.options[voiceSelect.selectedIndex].dataset.preview||'';voiceAudio.pause();voiceAudio.removeAttribute('src');if(url){{voiceAudio.src=url;voiceNote.textContent='点击播放可直接试听同一段对比文案。'}}else{{voiceNote.textContent='该 profile 还没有本地试听文件，可运行 tools/voice_audition.bat 生成。'}}voiceAudio.load()}});let timer;document.querySelectorAll('#script-form input,#script-form textarea,#script-form select').forEach(el=>el.addEventListener('input',()=>{{clearTimeout(timer);timer=setTimeout(()=>{{const body=new URLSearchParams(new FormData(document.getElementById('script-form')));body.set('action','autosave');fetch('/action',{{method:'POST',body}})}},900)}}));</script></main></body></html>"""


def _home(latest: dict[str, Any] | None, video_task: dict[str, Any] | None, csrf: str) -> str:
    draft_box = "<p>没有未完成脚本草稿。</p>"
    if latest:
        draft_box = f"<p><b>{_escape(latest.get('title') or '未命名')}</b> · {_escape(latest.get('mode'))} · {_escape(latest.get('updated_at'))}</p><button name=action value=resume_draft>恢复最近草稿</button>"
    task_box = "<p>没有失败或等待审核的视频任务。</p>"
    if video_task:
        task_box = f"<p><b>{_escape(video_task.get('project_id'))}</b> · {_escape(video_task.get('status'))} · {_escape(video_task.get('current_stage'))}</p><button name=action value=resume_video>按当前配置继续构建</button><p>配置未变时会续跑原任务；配置已变时会安全建立新任务，不会从错误阶段续跑。</p>"
    new_label = "放弃最近草稿并新建" if latest else "新建视频"
    return f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI Video</title><style>body{{max-width:850px;margin:50px auto;padding:20px;background:#101114;color:#f5f3ee;font:16px/1.6 system-ui}}section{{background:#1b1d22;padding:20px;border-radius:12px;margin:18px 0}}button{{background:#ffd54a;border:0;padding:12px 18px;border-radius:9px;font-weight:700;margin-right:10px}}</style></head><body><h1>AI 视频工作台</h1><form method=post action=/home><input type=hidden name=csrf value="{csrf}"><section><h2>未完成脚本</h2>{draft_box}<button name=action value=new>{new_label}</button></section><section><h2>视频任务</h2>{task_box}</section></form></body></html>"""


def run_workbench_server(
    *,
    resume_latest: bool = False,
    open_browser: bool = True,
    ready_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    csrf = secrets.token_urlsafe(24)
    latest = latest_editing_draft()
    current = latest if resume_latest and latest else None
    result: dict[str, Any] = {"status": "waiting", "project": None, "resume_task": None}
    server_holder: dict[str, ThreadingHTTPServer] = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(encoded)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _values(self) -> dict[str, list[str]] | None:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError:
                self._send(400, "bad request", "text/plain")
                return None
            if length < 0 or length > MAX_REQUEST_BYTES:
                self._send(413, "request too large", "text/plain")
                return None
            try:
                values = urllib.parse.parse_qs(
                    self.rfile.read(length).decode("utf-8"),
                    keep_blank_values=True,
                    max_num_fields=100,
                )
            except (UnicodeDecodeError, ValueError):
                self._send(400, "invalid form data", "text/plain")
                return None
            if values.get("csrf", [""])[0] != csrf:
                self._send(403, "forbidden", "text/plain")
                return None
            return values

        def do_GET(self) -> None:  # noqa: N802
            nonlocal current, latest
            if self.path.startswith("/voice-preview/"):
                profile = urllib.parse.unquote(self.path.removeprefix("/voice-preview/"))
                if not re.fullmatch(r"[a-z0-9_]+", profile):
                    self._send(404, "not found", "text/plain")
                    return
                preview = voice_preview_files().get(profile)
                if preview is None:
                    self._send(404, "not found", "text/plain")
                    return
                try:
                    audio = preview.read_bytes()
                except OSError:
                    self._send(404, "not found", "text/plain")
                    return
                self._send_bytes(200, audio, "audio/mpeg")
                return
            if self.path == "/":
                if current is not None:
                    self._send(200, _form(current, csrf))
                else:
                    latest = latest_editing_draft()
                    self._send(200, _home(latest, _latest_video_task(), csrf))
                return
            self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            nonlocal current, latest
            values = self._values()
            if values is None:
                return
            if self.path == "/home":
                action = values.get("action", [""])[0]
                if action == "resume_draft" and latest:
                    current = load_draft(str(latest["draft_id"]))
                    self._send(200, _form(current, csrf))
                    return
                if action == "new":
                    if latest:
                        discard_draft(load_draft(str(latest["draft_id"])))
                    current = create_draft()
                    self._send(200, _form(current, csrf))
                    return
                if action == "resume_video":
                    task = _latest_video_task()
                    if not task:
                        self._send(404, _home(latest, None, csrf), )
                        return
                    result["status"] = "resume_video"
                    result["resume_task"] = task
                    self._send(200, "<meta charset=utf-8><h2>正在继续视频任务，可以关闭此页面。</h2>")
                    threading.Thread(target=server_holder["server"].shutdown, daemon=True).start()
                    return
                self._send(400, "invalid action", "text/plain")
                return
            if self.path != "/action" or current is None:
                self._send(404, "not found", "text/plain")
                return
            action = values.get("action", [""])[0]
            try:
                if action in {"save", "autosave"}:
                    update_from_form(current, values)
                    if action == "save":
                        save_manual_version(
                            current,
                            values.get("feedback", ["手动保存"])[0] or "手动保存",
                        )
                    self._send(204 if action == "autosave" else 200, "" if action == "autosave" else _form(current, csrf, "草稿已保存。"))
                    return
                if action == "discard":
                    discard_draft(current)
                    current = create_draft()
                    self._send(200, _form(current, csrf, "旧草稿已放弃。"))
                    return
                update_from_form(current, values)
                if action == "restore":
                    token = values.get("version_to_restore", [""])[0]
                    kind, _, number_text = token.partition(":")
                    collection = current.get("versions", []) if kind == "ai" else current.get("manual_versions", []) if kind == "manual" else []
                    version = next((item for item in collection if str(item.get("version")) == number_text), None)
                    if not version:
                        raise ScriptWorkbenchError("请选择要恢复的历史版本。")
                    current["final_script"] = str(version["script"])
                    if kind == "ai":
                        current["analysis"] = version["analysis"]
                        current["suggestions"] = version["suggestions"]
                    save_draft(current)
                    self._send(200, _form(current, csrf, f"已恢复版本 {number_text} 到编辑区。"))
                    return
                if action == "ai":
                    mode = str(current.get("mode"))
                    if current.get("duration_seconds") not in ALLOWED_DURATIONS or not current.get("title"):
                        raise ScriptWorkbenchError("生成或审核前必须填写标题并选择时长。")
                    if mode == "direct":
                        raise ScriptWorkbenchError("direct 模式不会调用 AI；可直接编辑并锁稿。")
                    if mode == "review" and not current.get("original_script"):
                        raise ScriptWorkbenchError("review 模式必须先粘贴原稿。")
                    if mode == "topic" and not current.get("topic"):
                        raise ScriptWorkbenchError("topic 模式必须填写选题。")
                    run_ai_revision(current, values.get("feedback", [""])[0])
                    self._send(200, _form(current, csrf, "AI 版本已生成；请人工检查并编辑最终稿。"))
                    return
                if action == "lock":
                    project = lock_draft(current, values)
                    result["status"] = "locked"
                    result["project"] = str(project)
                    self._send(200, f"<meta charset=utf-8><h2>脚本已锁定：{_escape(project.name)}</h2><p>正在进入配音与视频流程，可以关闭此页面。</p>")
                    threading.Thread(target=server_holder["server"].shutdown, daemon=True).start()
                    return
                raise ScriptWorkbenchError("未知操作。")
            except ScriptWorkbenchError as exc:
                current["ai_error"] = str(exc)
                save_draft(current)
                self._send(400, _form(current, csrf, str(exc)))

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_holder["server"] = server
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[脚本工作台] 仅本机可访问：{url}")
    if ready_callback is not None:
        ready_callback(url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return result
