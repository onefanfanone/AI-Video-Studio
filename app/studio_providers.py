from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import yaml

from .comfyui_client import ComfyUIError, generate_image as generate_comfyui_image
from .deepseek_planner import DeepSeekPlannerError, request_json_object
from .openai_visuals import OpenAIVisualError, generate_image as generate_images_compatible
from .studio_profiles import ProfileError, validate_comfyui_workflow_profile
from .studio_settings import SecretStore, StudioPaths, get_studio_paths


class ProviderTestError(RuntimeError):
    """A redacted provider test failure safe to show in the local console."""


class ValidationStore:
    def __init__(self, paths: StudioPaths | None = None) -> None:
        self.paths = paths or get_studio_paths()
        self.path = self.paths.appdata_root / "state" / "profile-validations.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "profiles": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "profiles": {}}
        return payload if isinstance(payload, dict) else {"schema_version": 1, "profiles": {}}

    def record(self, kind: str, profile_id: str, fingerprint: str, result: Mapping[str, Any]) -> None:
        payload = self.load()
        payload.setdefault("profiles", {})[f"{kind}:{profile_id}"] = {
            "fingerprint": fingerprint,
            "tested_at": time.time(),
            "result": dict(result),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def status(self, kind: str, profile_id: str, fingerprint: str) -> dict[str, Any] | None:
        row = self.load().get("profiles", {}).get(f"{kind}:{profile_id}")
        if not isinstance(row, dict) or row.get("fingerprint") != fingerprint:
            return None
        return row


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    safe = {key: value for key, value in dict(profile).items() if key not in {"validated", "updated_at"}}
    return hashlib.sha256(
        json.dumps(safe, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _secret(profile: Mapping[str, Any], paths: StudioPaths) -> str:
    reference = str(profile.get("secret_ref") or "")
    if not reference:
        return ""
    value = SecretStore(paths.secrets_path).get(reference)
    if not value:
        raise ProviderTestError(f"密钥 {reference} 尚未配置。")
    return value


def _post_json(url: str, payload: Mapping[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Video-Studio/profile-test",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderTestError(f"模型测试返回 HTTP {exc.code}：{body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderTestError(f"模型测试连接失败：{type(exc).__name__}: {exc}") from exc


def test_llm_profile(profile: Mapping[str, Any], paths: StudioPaths | None = None) -> dict[str, Any]:
    studio_paths = paths or get_studio_paths()
    api_key = _secret(profile, studio_paths)
    protocol = str(profile.get("protocol"))
    started = time.monotonic()
    if protocol == "chat_completions":
        try:
            response, safe = request_json_object(
                "只输出 JSON 对象，顶层必须是 ok。",
                {"instruction": "Return {\"ok\": true}."},
                dict(profile),
                api_key,
            )
        except DeepSeekPlannerError as exc:
            raise ProviderTestError(str(exc)) from exc
        if response != {"ok": True}:
            raise ProviderTestError("模型 JSON 测试返回了错误结构。")
        model = safe.get("model")
    elif protocol == "openai_responses":
        endpoint = str(profile.get("base_url", "https://api.openai.com/v1")).rstrip("/") + "/responses"
        raw = _post_json(
            endpoint,
            {
                "model": profile.get("model"),
                "input": "Return exactly one JSON object: {\"ok\":true}",
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": min(128, int(profile.get("max_tokens", 128))),
            },
            api_key,
            float(profile.get("timeout_seconds", 120)),
        )
        text = "".join(
            str(part.get("text") or "")
            for item in raw.get("output", [])
            for part in item.get("content", [])
            if isinstance(part, dict)
        )
        try:
            response = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderTestError("Responses 测试没有返回有效 JSON。") from exc
        if response != {"ok": True}:
            raise ProviderTestError("Responses JSON 测试返回了错误结构。")
        model = raw.get("model")
    else:
        raise ProviderTestError(f"不支持的文本模型协议：{protocol}")
    result = {
        "status": "ok",
        "model": model or profile.get("model"),
        "latency_ms": round((time.monotonic() - started) * 1000),
    }
    ValidationStore(studio_paths).record(
        "llm", str(profile["id"]), profile_fingerprint(profile), result
    )
    return result


def test_comfyui_profile(
    image_profile: Mapping[str, Any],
    workflow_profile: Mapping[str, Any],
    paths: StudioPaths | None = None,
) -> tuple[bytes, dict[str, Any]]:
    studio_paths = paths or get_studio_paths()
    try:
        validation = validate_comfyui_workflow_profile(
            workflow_profile, image_profile, studio_paths
        )
        workflow_file = Path(str(workflow_profile.get("workflow_file") or ""))
        if not workflow_file.is_absolute():
            workflow_file = (
                studio_paths.workflow_root / workflow_file
                if (studio_paths.workflow_root / workflow_file).is_file()
                else studio_paths.code_root / workflow_file
            )
        config = {
            **dict(image_profile),
            "workflow_file": str(workflow_file),
            "prompt_marker": workflow_profile.get("prompt_marker", "__AI_VIDEO_PROMPT__"),
            "width": int(workflow_profile.get("width", 768)),
            "height": int(workflow_profile.get("height", 1344)),
            "timeout_seconds": min(900, int(image_profile.get("timeout_seconds", 900))),
        }
        image, metadata = generate_comfyui_image(
            "A simple museum artifact on a neutral background, vertical composition, no text, no watermark.",
            config,
            studio_paths.workspace,
            seed=1,
        )
    except (ComfyUIError, ProfileError) as exc:
        raise ProviderTestError(str(exc)) from exc
    result = {
        "status": "ok",
        "workflow_sha256": validation["workflow_sha256"],
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "size": metadata.get("size"),
    }
    validation_store = ValidationStore(studio_paths)
    validation_store.record(
        "image", str(image_profile["id"]), profile_fingerprint(image_profile), result
    )
    validation_store.record(
        "comfyui_workflow",
        str(workflow_profile["id"]),
        profile_fingerprint(workflow_profile),
        result,
    )
    return image, metadata


def test_external_image_profile(
    profile: Mapping[str, Any], paths: StudioPaths | None = None
) -> tuple[bytes, dict[str, Any]]:
    studio_paths = paths or get_studio_paths()
    api_key = _secret(profile, studio_paths)
    try:
        image, metadata = generate_images_compatible(
            "A plain clay amphora on a neutral museum background, vertical composition, no text, no watermark.",
            dict(profile),
            api_key,
        )
    except OpenAIVisualError as exc:
        raise ProviderTestError(str(exc)) from exc
    result = {
        "status": "ok",
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "model": metadata.get("model"),
        "size": metadata.get("size"),
    }
    ValidationStore(studio_paths).record(
        "image", str(profile["id"]), profile_fingerprint(profile), result
    )
    return image, metadata


async def _voice_audio(voice: str, rate: str, pitch: str, output: Path) -> None:
    import edge_tts

    communicator = edge_tts.Communicate(
        "如果你穿越到古罗马，街角那只陶罐，可能不是用来装水的。",
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicator.save(str(output))


def audition_voice_profile(
    profile: Mapping[str, Any], paths: StudioPaths | None = None
) -> Path:
    studio_paths = paths or get_studio_paths()
    fingerprint = profile_fingerprint(profile)
    output = studio_paths.cache_root / "voice-previews" / f"{profile['id']}-{fingerprint[:12]}.mp3"
    if output.is_file() and output.stat().st_size > 100:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp3")
    try:
        asyncio.run(
            _voice_audio(
                str(profile["voice"]),
                str(profile.get("rate", "+0%")),
                str(profile.get("pitch", "+0Hz")),
                temporary,
            )
        )
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ProviderTestError(f"Edge TTS 试听失败：{exc}") from exc
    os.replace(temporary, output)
    ValidationStore(studio_paths).record(
        "voice", str(profile["id"]), fingerprint, {"status": "ok"}
    )
    return output


def test_subtitle_profile(
    profile: Mapping[str, Any], paths: StudioPaths | None = None
) -> dict[str, Any]:
    studio_paths = paths or get_studio_paths()
    preset_id = str(profile.get("preset") or profile.get("id") or "")
    preset_file = studio_paths.code_root / "config" / "subtitle_presets.yaml"
    try:
        payload = yaml.safe_load(preset_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProviderTestError(f"找不到字幕预设文件：{preset_file}") from exc
    presets = payload.get("presets", {}) if isinstance(payload, dict) else {}
    if preset_id not in presets:
        raise ProviderTestError(f"字幕配置引用了未知预设：{preset_id}")
    settings = {**dict(presets[preset_id]), **dict(profile.get("settings") or {})}
    font_file = settings.get("font_file")
    if font_file:
        resolved_font = Path(os.path.expandvars(str(font_file))).expanduser().resolve()
        if not resolved_font.is_file():
            raise ProviderTestError(f"字幕字体文件不存在：{resolved_font}")
    else:
        font_name = str(settings.get("font_name") or "")
        candidates = list(studio_paths.font_root.glob("*")) + list(
            (studio_paths.runtime_root / "fonts").glob("*")
        )
        if "Source Han" in font_name and not any(
            "sourcehan" in item.name.lower() for item in candidates if item.is_file()
        ):
            raise ProviderTestError("未找到思源黑体字体文件，请先在环境页准备字体。")
    draft_font_name = str(settings.get("draft_font_type", "SourceHanSansCN_Bold"))
    try:
        import pyJianYingDraft as draft

        getattr(draft.FontType, draft_font_name)
    except (ImportError, AttributeError) as exc:
        raise ProviderTestError(f"剪映字体映射不可用：{draft_font_name}") from exc
    result = {
        "status": "ok",
        "font_name": settings.get("font_name"),
        "draft_font_type": draft_font_name,
    }
    ValidationStore(studio_paths).record(
        "subtitle", str(profile["id"]), profile_fingerprint(profile), result
    )
    return result
