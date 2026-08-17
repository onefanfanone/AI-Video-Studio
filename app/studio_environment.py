from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .comfyui_client import ComfyUIError, _request_json, validate_server_url
from .mpt_runtime import MPT_COMMIT
from .studio_settings import SettingsStore, StudioPaths, get_studio_paths


APP_VERSION = "0.6.2"


def _command(args: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return completed.returncode, completed.stdout.strip()


def _tool(name: str, version_args: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if not executable:
        return {"status": "missing", "path": None, "detail": "未找到"}
    code, output = _command([executable, *version_args])
    first = next((line.strip() for line in output.splitlines() if line.strip()), "")
    return {
        "status": "ok" if code == 0 else "warning",
        "path": executable,
        "detail": first[:240] or "已找到",
    }


def _comfy_status(url: str) -> dict[str, Any]:
    try:
        server = validate_server_url({"server_url": url})
        data = _request_json(f"{server}/object_info", timeout=3)
    except ComfyUIError as exc:
        return {"status": "offline", "path": url, "detail": str(exc)}
    return {
        "status": "ok",
        "path": url,
        "detail": f"已连接，发现 {len(data)} 种节点",
    }


def inspect_environment(paths: StudioPaths | None = None) -> dict[str, Any]:
    studio_paths = paths or get_studio_paths()
    settings = SettingsStore(studio_paths.appdata_root).load()
    runtime = studio_paths.runtime_root
    whisper_models = sorted(
        path.name for path in (runtime / "models").glob("whisper-*") if path.is_dir()
    ) if (runtime / "models").is_dir() else []
    mpt_marker = runtime / "mpt-venv" / f".moneyprinterturbo-{MPT_COMMIT}"
    font_candidates = list((runtime / "fonts").glob("*")) if (runtime / "fonts").is_dir() else []
    nvidia_path = shutil.which("nvidia-smi")
    if nvidia_path:
        code, output = _command(
            [
                nvidia_path,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        gpu = {
            "status": "ok" if code == 0 else "warning",
            "path": nvidia_path,
            "detail": output[:240] if output else "NVIDIA GPU",
        }
    else:
        gpu = {
            "status": "cpu",
            "path": None,
            "detail": "未发现 NVIDIA；Whisper 可使用 CPU/int8，但速度较慢",
        }
    comfy_url = "http://127.0.0.1:8000"
    try:
        from .studio_profiles import ProfileStore

        comfy_url = str(
            ProfileStore(studio_paths).get("image", "comfyui_default").get(
                "server_url", comfy_url
            )
        )
    except Exception:
        pass
    return {
        "schema_version": 1,
        "app_version": APP_VERSION,
        "platform": {
            "status": "ok" if os.name == "nt" and platform.machine().endswith("64") else "warning",
            "path": None,
            "detail": f"{platform.system()} {platform.release()} · {platform.machine()}",
        },
        "python": {
            "status": "ok" if sys.version_info[:2] == (3, 12) else "warning",
            "path": sys.executable,
            "detail": platform.python_version(),
        },
        "ffmpeg": _tool("ffmpeg", ["-version"]),
        "ffprobe": _tool("ffprobe", ["-version"]),
        "gpu": gpu,
        "jianying": {
            "status": "ok" if Path(str(settings["jianying_draft_root"])).is_dir() else "missing",
            "path": str(settings["jianying_draft_root"]),
            "detail": "草稿目录可用" if Path(str(settings["jianying_draft_root"])).is_dir() else "未找到草稿目录",
        },
        "moneyprinter": {
            "status": "ok" if mpt_marker.is_file() else "missing",
            "path": str(runtime / "MoneyPrinterTurbo"),
            "detail": "隔离运行时已就绪" if mpt_marker.is_file() else "需要手动准备或在向导中确认安装",
        },
        "whisper": {
            "status": "ok" if whisper_models else "missing",
            "path": str(runtime / "models"),
            "detail": "、".join(whisper_models) if whisper_models else "未发现本地 Whisper 模型（模型通常为数 GB）",
        },
        "fonts": {
            "status": "ok" if font_candidates else "missing",
            "path": str(runtime / "fonts"),
            "detail": f"发现 {len(font_candidates)} 个字体文件" if font_candidates else "未发现运行字体",
        },
        "comfyui": _comfy_status(comfy_url),
        "workspace": {
            "status": "ok" if studio_paths.workspace.is_dir() else "missing",
            "path": str(studio_paths.workspace),
            "detail": "旧版仓库兼容模式" if studio_paths.legacy_mode else "独立用户工作区",
        },
    }


def voice_cache_path(paths: StudioPaths | None = None) -> Path:
    return (paths or get_studio_paths()).cache_root / "edge-voices" / "voices.json"


async def _fetch_edge_voices() -> list[dict[str, Any]]:
    import edge_tts

    rows = await edge_tts.list_voices()
    result = []
    for item in rows:
        short_name = str(item.get("ShortName") or "")
        if not short_name:
            continue
        result.append(
            {
                "short_name": short_name,
                "friendly_name": str(item.get("FriendlyName") or short_name),
                "locale": str(item.get("Locale") or ""),
                "gender": str(item.get("Gender") or ""),
            }
        )
    return sorted(result, key=lambda row: (row["locale"], row["short_name"]))


def list_edge_voices(paths: StudioPaths | None = None, *, refresh: bool = False) -> list[dict[str, Any]]:
    path = voice_cache_path(paths)
    if path.is_file() and not refresh and time.time() - path.stat().st_mtime < 7 * 86400:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    try:
        rows = asyncio.run(_fetch_edge_voices())
    except Exception:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return rows


def check_release_once(settings_store: SettingsStore | None = None) -> dict[str, Any]:
    store = settings_store or SettingsStore()
    settings = store.load()
    update = dict(settings.get("updates") or {})
    if not update.get("enabled", True):
        return {"status": "disabled", "latest_version": None}
    last = update.get("last_checked_at")
    if last:
        try:
            if time.time() - float(last) < 86400:
                return {
                    "status": "cached",
                    "latest_version": update.get("latest_version"),
                }
        except (TypeError, ValueError):
            pass
    repository = str(update.get("repository") or "onefanfanone/AI-Video-Studio")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "AI-Video-Studio"},
    )
    latest = None
    status = "unavailable"
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latest = str(payload.get("tag_name") or "").lstrip("v") or None
        status = "ok"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        pass
    update["last_checked_at"] = time.time()
    update["latest_version"] = latest
    settings["updates"] = update
    store.save(settings)
    return {"status": status, "latest_version": latest}
