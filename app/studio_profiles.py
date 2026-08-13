from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .comfyui_client import ComfyUIError, load_api_workflow, prepare_workflow, validate_server_url
from .studio_settings import CODE_ROOT, StudioPaths, StudioSettingsError, get_studio_paths


PROFILE_SCHEMA_VERSION = 1
PROFILE_KINDS = ("llm", "image", "comfyui_workflow", "voice", "subtitle")
PROFILE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{1,63}")
SENSITIVE_KEYS = {
    "secret_ref",
    "credential_id",
    "api_key",
    "token",
    "authorization",
}


class ProfileError(RuntimeError):
    """A configuration-profile error safe to show in the local console."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _yaml_catalog(path: Path, key: str) -> dict[str, dict[str, Any]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    values = payload.get(key, {}) if isinstance(payload, dict) else {}
    return {
        str(profile_id): dict(value)
        for profile_id, value in values.items()
        if isinstance(value, dict)
    }


def builtin_profiles(code_root: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    root = (code_root or CODE_ROOT).resolve()
    voice_path = root / "config" / "voice_profiles.yaml"
    subtitle_path = root / "config" / "subtitle_presets.yaml"
    voices = _yaml_catalog(
        voice_path if voice_path.is_file() else CODE_ROOT / "config" / "voice_profiles.yaml",
        "profiles",
    )
    subtitles = _yaml_catalog(
        subtitle_path if subtitle_path.is_file() else CODE_ROOT / "config" / "subtitle_presets.yaml",
        "presets",
    )
    voice_profiles = {
        profile_id: {
            "schema_version": 1,
            "kind": "voice",
            "id": profile_id,
            "name": str(value.get("label") or profile_id),
            "builtin": True,
            "provider": "edge_tts",
            "voice": value.get("voice"),
            "rate": value.get("rate", "+0%"),
            "pitch": value.get("pitch", "+0Hz"),
            "version": 1,
        }
        for profile_id, value in voices.items()
    }
    subtitle_profiles = {
        profile_id: {
            "schema_version": 1,
            "kind": "subtitle",
            "id": profile_id,
            "name": {
                "history_clean": "历史简洁白字",
                "history_keyword": "历史金色关键词",
                "history_hook": "历史钩子轻入场",
                "social_pink": "社交粉色粗书体",
            }.get(profile_id, profile_id),
            "builtin": True,
            "preset": profile_id,
            "settings": value,
            "version": 1,
        }
        for profile_id, value in subtitles.items()
    }
    return {
        "llm": {
            "deepseek_default": {
                "schema_version": 1,
                "kind": "llm",
                "id": "deepseek_default",
                "name": "DeepSeek 默认",
                "builtin": True,
                "protocol": "chat_completions",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
                "json_mode": "response_format",
                "thinking": "disabled",
                "timeout_seconds": 120,
                "max_tokens": 12000,
                "secret_ref": "DEEPSEEK_API_KEY",
                "validated": False,
                "version": 1,
            },
            "openai_responses": {
                "schema_version": 1,
                "kind": "llm",
                "id": "openai_responses",
                "name": "OpenAI Responses",
                "builtin": True,
                "protocol": "openai_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "timeout_seconds": 120,
                "max_tokens": 12000,
                "secret_ref": "OPENAI_API_KEY",
                "validated": False,
                "version": 1,
            },
        },
        "image": {
            "comfyui_default": {
                "schema_version": 1,
                "kind": "image",
                "id": "comfyui_default",
                "name": "本机 ComfyUI",
                "builtin": True,
                "protocol": "comfyui_local",
                "server_url": "http://127.0.0.1:8000",
                "timeout_seconds": 900,
                "validated": False,
                "version": 1,
            },
            "openai_images": {
                "schema_version": 1,
                "kind": "image",
                "id": "openai_images",
                "name": "OpenAI Images",
                "builtin": True,
                "protocol": "images_compatible",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-image-2",
                "size": "1024x1536",
                "quality": "medium",
                "output_format": "jpeg",
                "max_images_per_run": 4,
                "secret_ref": "OPENAI_API_KEY",
                "validated": False,
                "version": 1,
            },
        },
        "comfyui_workflow": {
            "history_image_default": {
                "schema_version": 1,
                "kind": "comfyui_workflow",
                "id": "history_image_default",
                "name": "历史竖屏 · Z-Image Turbo",
                "builtin": True,
                "workflow_file": "history_image_api.json",
                "prompt_marker": "__AI_VIDEO_PROMPT__",
                "width": 768,
                "height": 1344,
                "bindings": {
                    "prompt": ["57:27", "text"],
                    "seed": ["57:3", "seed"],
                    "width": ["57:13", "width"],
                    "height": ["57:13", "height"],
                    "output": ["9", "images"],
                },
                "validated": False,
                "version": 1,
            }
        },
        "voice": voice_profiles,
        "subtitle": subtitle_profiles,
    }


def _profile_path(paths: StudioPaths, kind: str, profile_id: str) -> Path:
    if kind not in PROFILE_KINDS:
        raise ProfileError(f"未知配置档类型：{kind}")
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ProfileError("配置档 ID 只能使用小写字母、数字、短横线和下划线。")
    return paths.profile_root / kind / f"{profile_id}.json"


def validate_profile(profile: Mapping[str, Any], expected_kind: str | None = None) -> dict[str, Any]:
    result = deepcopy(dict(profile))
    kind = str(result.get("kind") or expected_kind or "")
    profile_id = str(result.get("id") or "")
    if kind not in PROFILE_KINDS:
        raise ProfileError(f"未知配置档类型：{kind}")
    if expected_kind and kind != expected_kind:
        raise ProfileError(f"配置档类型应为 {expected_kind}，收到 {kind}。")
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ProfileError("配置档缺少合法 ID。")
    if not str(result.get("name") or "").strip():
        raise ProfileError("配置档名称不能为空。")
    if kind == "llm":
        protocol = str(result.get("protocol") or "")
        if protocol not in {"chat_completions", "openai_responses"}:
            raise ProfileError("文本模型协议只能是 chat_completions 或 openai_responses。")
        if not str(result.get("base_url") or "").startswith("https://"):
            raise ProfileError("文本模型 Base URL 必须使用 HTTPS。")
        if not str(result.get("model") or "").strip():
            raise ProfileError("文本模型 ID 不能为空。")
        timeout = int(result.get("timeout_seconds", 120))
        max_tokens = int(result.get("max_tokens", 12000))
        if not 10 <= timeout <= 900 or not 128 <= max_tokens <= 100000:
            raise ProfileError("文本模型超时或最大输出范围无效。")
    elif kind == "image":
        protocol = str(result.get("protocol") or "")
        if protocol not in {"comfyui_local", "images_compatible"}:
            raise ProfileError("生图协议只能是 comfyui_local 或 images_compatible。")
        if protocol == "comfyui_local":
            try:
                validate_server_url(result)
            except ComfyUIError as exc:
                raise ProfileError(str(exc)) from exc
        else:
            if not str(result.get("base_url") or "").startswith("https://"):
                raise ProfileError("外部生图 Base URL 必须使用 HTTPS。")
            limit = result.get("max_images_per_run")
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
                raise ProfileError("外部生图必须配置 1–1000 的固定单次上限。")
    elif kind == "voice":
        if result.get("provider", "edge_tts") != "edge_tts":
            raise ProfileError("首发版语音只支持 Edge TTS。")
        if not str(result.get("voice") or "").strip():
            raise ProfileError("Edge TTS voice 不能为空。")
        if not re.fullmatch(r"[+-]\d{1,3}%", str(result.get("rate", "+0%"))):
            raise ProfileError("语速必须使用 +0% 形式。")
        if not re.fullmatch(r"[+-]\d{1,3}Hz", str(result.get("pitch", "+0Hz"))):
            raise ProfileError("音高必须使用 +0Hz 形式。")
    elif kind == "subtitle":
        settings = result.get("settings")
        if not isinstance(settings, dict):
            raise ProfileError("字幕配置档缺少 settings。")
        chars = int(settings.get("max_chars_per_line", 8))
        lines = int(settings.get("max_lines", 1))
        if not 1 <= chars <= 24 or not 1 <= lines <= 2:
            raise ProfileError("字幕每行字数必须为 1–24，行数必须为 1–2。")
    result["schema_version"] = PROFILE_SCHEMA_VERSION
    result["kind"] = kind
    result["id"] = profile_id
    result.setdefault("version", 1)
    result.setdefault("validated", False)
    return result


class ProfileStore:
    def __init__(self, paths: StudioPaths | None = None) -> None:
        self.paths = paths or get_studio_paths()

    def list(self, kind: str) -> dict[str, dict[str, Any]]:
        catalog = deepcopy(builtin_profiles(self.paths.code_root).get(kind, {}))
        folder = self.paths.profile_root / kind
        if folder.is_dir():
            for path in sorted(folder.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    valid = validate_profile(payload, kind)
                except (OSError, json.JSONDecodeError, ProfileError):
                    continue
                catalog[valid["id"]] = valid
        return catalog

    def get(self, kind: str, profile_id: str) -> dict[str, Any]:
        catalog = self.list(kind)
        if profile_id not in catalog:
            raise ProfileError(f"找不到 {kind} 配置档：{profile_id}")
        return deepcopy(catalog[profile_id])

    def save(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        valid = validate_profile(profile)
        path = _profile_path(self.paths, valid["kind"], valid["id"])
        existing = self.list(valid["kind"]).get(valid["id"])
        if existing and existing.get("builtin"):
            raise ProfileError("内置配置档不可覆盖；请另存为新的配置档。")
        valid["builtin"] = False
        valid["updated_at"] = _now()
        valid["version"] = int((existing or {}).get("version", 0)) + 1
        _atomic_json(path, valid)
        return valid

    def snapshot(self, selections: Mapping[str, str]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for role, token in selections.items():
            kind, separator, profile_id = str(token).partition(":")
            if not separator or kind not in PROFILE_KINDS:
                raise ProfileError(f"配置选择格式错误：{role}")
            resolved[role] = self.get(kind, profile_id)
        snapshot = {
            "schema_version": 1,
            "created_at": _now(),
            "profiles": resolved,
        }
        snapshot["sha256"] = snapshot_hash(snapshot)
        return snapshot


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(snapshot))
    payload.pop("sha256", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def scan_workflow(workflow: Mapping[str, Any], marker: str = "__AI_VIDEO_PROMPT__") -> dict[str, Any]:
    prompt: list[list[str]] = []
    seeds: list[list[str]] = []
    widths: list[list[str]] = []
    heights: list[list[str]] = []
    outputs: list[list[str]] = []
    models: dict[str, list[dict[str, str]]] = {
        "checkpoint": [],
        "unet": [],
        "clip": [],
        "vae": [],
        "lora": [],
    }
    samplers: list[dict[str, Any]] = []
    model_keys = {
        "ckpt_name": "checkpoint",
        "unet_name": "unet",
        "clip_name": "clip",
        "vae_name": "vae",
        "lora_name": "lora",
    }
    for node_id, raw_node in workflow.items():
        if not isinstance(raw_node, dict) or not isinstance(raw_node.get("inputs"), dict):
            raise ProfileError(f"工作流节点 {node_id} 不是 ComfyUI API 格式。")
        node = dict(raw_node)
        class_type = str(node.get("class_type") or "")
        if not class_type:
            raise ProfileError(f"工作流节点 {node_id} 缺少 class_type。")
        inputs = node["inputs"]
        for input_name, value in inputs.items():
            if isinstance(value, str) and marker in value:
                prompt.append([str(node_id), str(input_name)])
            if input_name in {"seed", "noise_seed"} and isinstance(value, int):
                seeds.append([str(node_id), str(input_name)])
            if input_name == "width" and isinstance(value, int):
                widths.append([str(node_id), str(input_name)])
            if input_name == "height" and isinstance(value, int):
                heights.append([str(node_id), str(input_name)])
            if input_name in model_keys and isinstance(value, str):
                models[model_keys[input_name]].append(
                    {"node_id": str(node_id), "input": str(input_name), "value": value}
                )
        if class_type == "SaveImage":
            outputs.append([str(node_id), "images"])
        if class_type in {"KSampler", "KSamplerAdvanced"}:
            samplers.append(
                {
                    "node_id": str(node_id),
                    **{
                        key: inputs[key]
                        for key in ("steps", "cfg", "sampler_name", "scheduler", "denoise")
                        if key in inputs
                    },
                }
            )
    bindings = {
        "prompt": prompt[0] if len(prompt) == 1 else None,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "width": widths[0] if len(widths) == 1 else None,
        "height": heights[0] if len(heights) == 1 else None,
        "output": outputs[0] if len(outputs) == 1 else None,
    }
    unresolved = [name for name, value in bindings.items() if value is None]
    return {
        "bindings": bindings,
        "unresolved": unresolved,
        "candidates": {
            "prompt": prompt,
            "seed": seeds,
            "width": widths,
            "height": heights,
            "output": outputs,
        },
        "models": models,
        "samplers": samplers,
        "node_count": len(workflow),
    }


def import_workflow_profile(
    source: Path,
    profile_id: str,
    name: str,
    *,
    paths: StudioPaths | None = None,
    marker: str = "__AI_VIDEO_PROMPT__",
    bindings: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    studio_paths = paths or get_studio_paths()
    if not source.is_file():
        raise ProfileError(f"找不到工作流文件：{source}")
    try:
        workflow = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError("工作流不是有效 JSON。") from exc
    if not isinstance(workflow, dict) or not workflow:
        raise ProfileError("工作流必须是 ComfyUI API 格式 JSON 对象。")
    scan = scan_workflow(workflow, marker)
    destination = studio_paths.workflow_root / f"{profile_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    resolved_bindings = dict(scan["bindings"])
    resolved_bindings.update(dict(bindings or {}))
    unresolved = [key for key, value in resolved_bindings.items() if not value]
    profile = {
        "schema_version": 1,
        "kind": "comfyui_workflow",
        "id": profile_id,
        "name": name.strip(),
        "workflow_file": destination.name,
        "prompt_marker": marker,
        "width": 768,
        "height": 1344,
        "bindings": resolved_bindings,
        "scan": scan,
        "validated": not unresolved,
        "test_image_completed": False,
    }
    return ProfileStore(studio_paths).save(profile)


def validate_comfyui_workflow_profile(
    profile: Mapping[str, Any],
    image_profile: Mapping[str, Any],
    paths: StudioPaths | None = None,
) -> dict[str, Any]:
    studio_paths = paths or get_studio_paths()
    workflow_file = Path(str(profile.get("workflow_file") or ""))
    if not workflow_file.is_absolute():
        candidates = [
            studio_paths.workflow_root / workflow_file,
            studio_paths.code_root / workflow_file,
        ]
        workflow_file = next((item for item in candidates if item.is_file()), candidates[0])
    config = {
        **dict(image_profile),
        "workflow_file": str(workflow_file),
        "prompt_marker": profile.get("prompt_marker", "__AI_VIDEO_PROMPT__"),
        "width": int(profile.get("width", 768)),
        "height": int(profile.get("height", 1344)),
    }
    validate_server_url(config)
    workflow, _ = load_api_workflow(config, studio_paths.workspace)
    prepared = prepare_workflow(workflow, "AI-Video Studio workflow test", config, 1)
    return {
        "valid": True,
        "workflow_sha256": hashlib.sha256(
            json.dumps(prepared, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "scan": scan_workflow(workflow, str(config["prompt_marker"])),
    }


def _scrub_export(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_KEYS:
                continue
            if lowered.endswith("_path") or lowered.endswith("_root"):
                continue
            if lowered in {"workflow_file", "font_file"} and Path(str(child)).is_absolute():
                continue
            result[str(key)] = _scrub_export(child)
        return result
    if isinstance(value, list):
        return [_scrub_export(item) for item in value]
    return value


def export_profile_bundle(destination: Path, store: ProfileStore | None = None) -> Path:
    profile_store = store or ProfileStore()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai-video-profile-export-") as folder:
        root = Path(folder)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "exported_at": _now(),
            "requires_secret_rebinding": True,
            "profiles": [],
        }
        for kind in PROFILE_KINDS:
            for profile in profile_store.list(kind).values():
                if profile.get("builtin"):
                    continue
                clean = _scrub_export(profile)
                clean["validated"] = False
                path = root / "profiles" / kind / f"{profile['id']}.json"
                _atomic_json(path, clean)
                manifest["profiles"].append({"kind": kind, "id": profile["id"]})
        _atomic_json(root / "manifest.json", manifest)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return destination


def import_profile_bundle(source: Path, store: ProfileStore | None = None) -> list[dict[str, Any]]:
    profile_store = store or ProfileStore()
    imported: list[dict[str, Any]] = []
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise ProfileError("配置包缺少 manifest.json。")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ProfileError("配置包包含不安全路径。")
            if not name.startswith("profiles/") or not name.endswith(".json"):
                continue
            payload = json.loads(archive.read(name).decode("utf-8"))
            payload["validated"] = False
            imported.append(profile_store.save(payload))
    return imported
