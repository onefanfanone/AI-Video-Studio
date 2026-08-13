from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_SCHEMA_VERSION = 1
WORKSPACE_DIRS = (
    "projects",
    "outputs",
    "raw",
    "cache",
    "runtime",
    "workflows",
    "profiles",
    "fonts",
    "exports",
)
SECRET_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,80}")


class StudioSettingsError(RuntimeError):
    """A settings or credential failure safe to show in the local console."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def studio_data_root() -> Path:
    override = os.environ.get("AI_VIDEO_APPDATA", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if not local:
        local = str(Path.home() / "AppData" / "Local")
    return (Path(local) / "AI-Video-Studio").resolve()


def default_workspace() -> Path:
    override = os.environ.get("AI_VIDEO_WORKSPACE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Videos" / "AI-Video-Studio").resolve()


def discover_jianying_draft_root() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    return (
        local
        / "JianyingPro"
        / "User Data"
        / "Projects"
        / "com.lveditor.draft"
    ).resolve()


def default_settings() -> dict[str, Any]:
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "initialized": False,
        "workspace": str(default_workspace()),
        "jianying_draft_root": str(discover_jianying_draft_root()),
        "toolchain": {
            "python": "auto",
            "ffmpeg": "auto",
            "whisper_device": "auto",
        },
        "defaults": {
            "llm": "deepseek_default",
            "script_llm": "deepseek_default",
            "visual_llm": "deepseek_default",
            "semantic_llm": "deepseek_default",
            "image": "comfyui_default",
            "comfyui_workflow": "history_image_default",
            "voice": "yunyang_soft",
            "subtitle": "social_pink",
            "visual_strategy": "museum_and_ai",
            "candidates_per_shot": 4,
        },
        "updates": {
            "enabled": True,
            "repository": "onefanfanone/AI-Video-Studio",
            "channel": "stable",
            "last_checked_at": None,
            "latest_version": None,
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


class SettingsStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or studio_data_root()).resolve()
        self.path = self.root / "settings.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return default_settings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudioSettingsError(f"无法读取设置文件：{self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise StudioSettingsError("settings.json 不是受支持的 schema v1。")
        merged = default_settings()
        merged.update(payload)
        for key in ("toolchain", "defaults", "updates"):
            value = dict(merged.get(key) or {})
            value.update(payload.get(key) or {})
            merged[key] = value
        return merged

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        settings = dict(payload)
        settings["schema_version"] = SETTINGS_SCHEMA_VERSION
        settings["updated_at"] = _now()
        workspace = Path(str(settings.get("workspace") or default_workspace())).expanduser()
        if not workspace.is_absolute():
            raise StudioSettingsError("工作区必须使用绝对路径。")
        settings["workspace"] = str(workspace.resolve())
        draft_root = Path(
            str(settings.get("jianying_draft_root") or discover_jianying_draft_root())
        ).expanduser()
        if not draft_root.is_absolute():
            raise StudioSettingsError("剪映草稿目录必须使用绝对路径。")
        settings["jianying_draft_root"] = str(draft_root.resolve())
        _atomic_json(self.path, settings)
        return settings

    def initialize(
        self,
        workspace: Path,
        *,
        jianying_draft_root: Path | None = None,
    ) -> dict[str, Any]:
        settings = self.load()
        settings["workspace"] = str(workspace.expanduser().resolve())
        settings["jianying_draft_root"] = str(
            (jianying_draft_root or discover_jianying_draft_root()).expanduser().resolve()
        )
        settings["initialized"] = True
        ensure_workspace(Path(settings["workspace"]))
        return self.save(settings)


@dataclass(frozen=True)
class StudioPaths:
    code_root: Path
    appdata_root: Path
    workspace: Path
    settings_path: Path
    secrets_path: Path
    legacy_mode: bool = False

    @property
    def project_root(self) -> Path:
        return self.workspace / "projects"

    @property
    def output_root(self) -> Path:
        return self.workspace / "outputs"

    @property
    def raw_root(self) -> Path:
        return self.workspace / "raw"

    @property
    def cache_root(self) -> Path:
        return self.workspace / "cache"

    @property
    def runtime_root(self) -> Path:
        return self.workspace / "runtime"

    @property
    def workflow_root(self) -> Path:
        return self.workspace / "workflows"

    @property
    def profile_root(self) -> Path:
        return self.workspace / "profiles"

    @property
    def font_root(self) -> Path:
        return self.workspace / "fonts"

    @property
    def export_root(self) -> Path:
        return self.workspace / "exports"

    @property
    def draft_root(self) -> Path:
        return self.project_root / "_drafts"


def ensure_workspace(workspace: Path) -> None:
    resolved = workspace.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_DIRS:
        (resolved / name).mkdir(parents=True, exist_ok=True)


def get_studio_paths(code_root: Path | None = None) -> StudioPaths:
    code = (code_root or CODE_ROOT).resolve()
    appdata = studio_data_root()
    store = SettingsStore(appdata)
    settings = store.load()
    force_workspace = os.environ.get("AI_VIDEO_WORKSPACE", "").strip()
    # Patched roots in the existing unit tests retain the original repository layout.
    use_legacy = code != CODE_ROOT.resolve() and not force_workspace
    initialized = bool(settings.get("initialized")) or bool(force_workspace)
    if use_legacy or not initialized:
        workspace = code
        legacy = True
    else:
        workspace = Path(str(settings["workspace"])).expanduser().resolve()
        legacy = False
    return StudioPaths(
        code_root=code,
        appdata_root=appdata,
        workspace=workspace,
        settings_path=store.path,
        secrets_path=appdata / "secrets.dat",
        legacy_mode=legacy,
    )


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data)
    return (
        _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))),
        buffer,
    )


def dpapi_encrypt(data: bytes, *, entropy: bytes = b"AI-Video-Studio/v1") -> bytes:
    if os.name != "nt":
        raise StudioSettingsError("secrets.dat 只支持 Windows DPAPI。")
    source, source_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(entropy)
    output = _DataBlob()
    del source_buffer, entropy_buffer
    result = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source),
        "AI-Video-Studio",
        ctypes.byref(entropy_blob),
        None,
        None,
        0x1,
        ctypes.byref(output),
    )
    if not result:
        raise StudioSettingsError("Windows DPAPI 加密失败。")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def dpapi_decrypt(data: bytes, *, entropy: bytes = b"AI-Video-Studio/v1") -> bytes:
    if os.name != "nt":
        raise StudioSettingsError("secrets.dat 只支持 Windows DPAPI。")
    source, source_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(entropy)
    output = _DataBlob()
    del source_buffer, entropy_buffer
    result = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        0x1,
        ctypes.byref(output),
    )
    if not result:
        raise StudioSettingsError("secrets.dat 无法由当前 Windows 用户解密。")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


class SecretStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or (studio_data_root() / "secrets.dat")).resolve()

    def _load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            decrypted = dpapi_decrypt(self.path.read_bytes())
            payload = json.loads(decrypted.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StudioSettingsError("secrets.dat 已损坏或不属于当前 Windows 用户。") from exc
        if not isinstance(payload, dict):
            raise StudioSettingsError("secrets.dat 内容格式错误。")
        return {str(key): str(value) for key, value in payload.items() if str(value)}

    def _save(self, payload: Mapping[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
        encrypted = dpapi_encrypt(raw)
        temporary = self.path.with_suffix(".dat.tmp")
        temporary.write_bytes(encrypted)
        os.replace(temporary, self.path)

    def set(self, name: str, value: str) -> None:
        if not SECRET_NAME_RE.fullmatch(name):
            raise StudioSettingsError("密钥引用名必须是大写字母、数字和下划线。")
        clean = value.strip()
        if not clean:
            raise StudioSettingsError("密钥不能为空。")
        payload = self._load()
        payload[name] = clean
        self._save(payload)
        if self.get(name) != clean:
            raise StudioSettingsError("密钥写入后的 DPAPI 往返验证失败。")

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def delete(self, name: str) -> None:
        payload = self._load()
        payload.pop(name, None)
        self._save(payload)

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "configured": True,
                "fingerprint": hashlib.sha256(value.encode("utf-8")).hexdigest()[:8],
            }
            for key, value in self._load().items()
        }

    def as_environment(self) -> dict[str, str]:
        return self._load()


def read_legacy_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if SECRET_NAME_RE.fullmatch(key) and value:
            values[key] = value
    return values


def import_legacy_env(path: Path, store: SecretStore | None = None) -> list[str]:
    target = store or SecretStore()
    imported: list[str] = []
    for key, value in read_legacy_env(path).items():
        target.set(key, value)
        imported.append(key)
    return imported


def load_runtime_secrets(code_root: Path | None = None) -> dict[str, str]:
    code = (code_root or CODE_ROOT).resolve()
    values = read_legacy_env(code / ".env.local")
    try:
        values.update(SecretStore(get_studio_paths(code).secrets_path).as_environment())
    except StudioSettingsError:
        # The UI environment check reports the DPAPI failure without leaking values.
        pass
    for key, value in os.environ.items():
        if SECRET_NAME_RE.fullmatch(key) and (
            key.endswith("_API_KEY") or key.endswith("_TOKEN")
        ):
            values[key] = value
    return values


def public_settings(settings: Mapping[str, Any], secret_status: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only settings representation permitted in HTTP responses."""
    return {
        **dict(settings),
        "secrets": {key: {"configured": True} for key in sorted(secret_status)},
    }
