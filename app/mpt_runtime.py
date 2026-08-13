from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .studio_settings import get_studio_paths


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = get_studio_paths(ROOT).runtime_root
MPT_VERSION = "1.3.3"
MPT_COMMIT = "b4218dd66851acf2e19d4aa5f10252b08380f742"
MPT_ARCHIVE_URL = (
    "https://github.com/harry0703/MoneyPrinterTurbo/archive/"
    f"{MPT_COMMIT}.zip"
)
MPT_SOURCE = RUNTIME_ROOT / "MoneyPrinterTurbo"
MPT_VENV = RUNTIME_ROOT / "mpt-venv"
MPT_MARKER = MPT_VENV / f".moneyprinterturbo-{MPT_COMMIT}"
CUDA_RUNTIME_MARKER = MPT_VENV / ".cuda12-runtime-v1"
CUDA_RUNTIME_PACKAGES = (
    "nvidia-cublas-cu12==12.9.2.10",
    "nvidia-cudnn-cu12==9.11.0.98",
)

SOURCE_HAN_VERSION = "2.005R"
SOURCE_HAN_URL = (
    "https://github.com/adobe-fonts/source-han-sans/releases/download/"
    f"{SOURCE_HAN_VERSION}/09_SourceHanSansSC.zip"
)
SOURCE_HAN_SHA256 = "ef7364f7ac2564be1ae9c1d74276de2653fe38b73449070398c4fc0b7e032ff1"
SOURCE_HAN_FONT = RUNTIME_ROOT / "fonts" / "SourceHanSansSC-Heavy.otf"


def configure_runtime_root(path: Path) -> None:
    """Refresh derived runtime locations after the first-run wizard changes workspace."""
    global RUNTIME_ROOT, MPT_SOURCE, MPT_VENV, MPT_MARKER, CUDA_RUNTIME_MARKER, SOURCE_HAN_FONT
    RUNTIME_ROOT = path.expanduser().resolve()
    MPT_SOURCE = RUNTIME_ROOT / "MoneyPrinterTurbo"
    MPT_VENV = RUNTIME_ROOT / "mpt-venv"
    MPT_MARKER = MPT_VENV / f".moneyprinterturbo-{MPT_COMMIT}"
    CUDA_RUNTIME_MARKER = MPT_VENV / ".cuda12-runtime-v1"
    SOURCE_HAN_FONT = RUNTIME_ROOT / "fonts" / "SourceHanSansSC-Heavy.otf"


class RuntimeSetupError(RuntimeError):
    pass


def _python_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(args: Sequence[str | Path], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(item) for item in args],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeSetupError(detail[-5000:] or "未知子进程错误")
    return completed


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Video-Builder/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeSetupError(f"下载失败：{url}\n{exc}") from exc
    os.replace(temporary, destination)


def _safe_remove_runtime_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != RUNTIME_ROOT.resolve():
        raise RuntimeSetupError(f"拒绝清理运行目录之外的路径：{resolved}")
    def remove_readonly(function: Any, target: str, error: BaseException) -> None:
        del error
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(resolved, ignore_errors=False, onexc=remove_readonly)


def _ensure_mpt_source() -> None:
    version_file = MPT_SOURCE / ".ai-video-version.json"
    if version_file.is_file():
        try:
            payload = json.loads(version_file.read_text(encoding="utf-8"))
            if payload.get("commit") == MPT_COMMIT and (MPT_SOURCE / "pyproject.toml").is_file():
                return
        except (OSError, json.JSONDecodeError):
            pass
    if MPT_SOURCE.exists():
        _safe_remove_runtime_dir(MPT_SOURCE)

    archive = RUNTIME_ROOT / f"MoneyPrinterTurbo-{MPT_COMMIT}.zip"
    _download(MPT_ARCHIVE_URL, archive)
    extract_root = RUNTIME_ROOT / f"mpt-extract-{MPT_COMMIT}"
    if extract_root.exists():
        _safe_remove_runtime_dir(extract_root)
    extract_root.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as package:
            package.extractall(extract_root)
        folders = [item for item in extract_root.iterdir() if item.is_dir()]
        if len(folders) != 1 or not (folders[0] / "pyproject.toml").is_file():
            raise RuntimeSetupError("MoneyPrinterTurbo 压缩包结构不符合预期。")
        folders[0].replace(MPT_SOURCE)
        version_file.write_text(
            json.dumps(
                {"version": MPT_VERSION, "commit": MPT_COMMIT, "source": MPT_ARCHIVE_URL},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if extract_root.exists():
            _safe_remove_runtime_dir(extract_root)
        archive.unlink(missing_ok=True)


def _nvidia_bin_dirs() -> list[Path]:
    site_packages = MPT_VENV / "Lib" / "site-packages"
    return [
        directory
        for directory in (
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "nvidia" / "cudnn" / "bin",
        )
        if directory.is_dir()
    ]


def ensure_mpt_runtime(*, with_cuda: bool = False) -> Path:
    """Install the pinned MPT application in a separate Python 3.12 environment."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeSetupError("MoneyPrinterTurbo 隔离环境必须由 Python 3.12 创建。")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_mpt_source()
    python = _python_in_venv(MPT_VENV)
    if not python.is_file():
        _run([sys.executable, "-m", "venv", MPT_VENV])
    if not MPT_MARKER.is_file():
        print("[准备] 首次安装 MoneyPrinterTurbo 本阶段隔离依赖，可能需要数分钟...")
        # MPT is an application with WebUI/API/LLM dependencies that this bridge never imports.
        # Install the exact TTS/alignment versions declared by v1.3.3 instead of pulling the
        # unrelated application stack. This keeps the top-level `app` packages isolated and
        # makes the first local setup tractable.
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "edge-tts==7.2.7",
                "faster-whisper==1.1.0",
            ]
        )
        _run([python, "-c", "import edge_tts, faster_whisper; print('runtime ok')"])
        MPT_MARKER.write_text(MPT_COMMIT + "\n", encoding="utf-8")
    if with_cuda and not CUDA_RUNTIME_MARKER.is_file():
        print("[准备] 安装隔离的 CUDA 12 cuBLAS/cuDNN 运行库（约 1.1 GB）...")
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *CUDA_RUNTIME_PACKAGES,
            ]
        )
        if len(_nvidia_bin_dirs()) != 2:
            raise RuntimeSetupError("CUDA/cuDNN Python 包已安装，但找不到 Windows DLL 目录。")
        CUDA_RUNTIME_MARKER.write_text("\n".join(CUDA_RUNTIME_PACKAGES) + "\n", encoding="utf-8")
    return python


def invoke_worker(request: dict[str, Any], response_path: Path) -> dict[str, Any]:
    needs_cuda = request.get("operation") == "align" and request.get("device") == "cuda"
    python = ensure_mpt_runtime(with_cuda=needs_cuda)
    request_path = response_path.with_suffix(".request.json")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        **request,
        "moneyprinterturbo": {"version": MPT_VERSION, "commit": MPT_COMMIT},
    }
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    response_path.unlink(missing_ok=True)
    worker_env = os.environ.copy()
    worker_env.setdefault("HF_HOME", str(RUNTIME_ROOT / "huggingface"))
    worker_env.setdefault("HF_HUB_DISABLE_XET", "1")
    nvidia_path = os.pathsep.join(str(path) for path in _nvidia_bin_dirs())
    if nvidia_path:
        worker_env["PATH"] = nvidia_path + os.pathsep + worker_env.get("PATH", "")
    completed = subprocess.run(
        [python, ROOT / "app" / "mpt_worker.py", "--request", request_path, "--response", response_path],
        cwd=str(MPT_SOURCE),
        env=worker_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if not response_path.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeSetupError(f"MoneyPrinterTurbo worker 未返回结果。\n{detail[-5000:]}")
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    if completed.returncode or payload.get("status") != "success":
        raise RuntimeSetupError(str(payload.get("error") or completed.stderr or "worker 失败"))
    return payload


def ensure_source_han_font(*, auto_download: bool = True) -> Path | None:
    if SOURCE_HAN_FONT.is_file() and SOURCE_HAN_FONT.stat().st_size > 1_000_000:
        return SOURCE_HAN_FONT
    if not auto_download:
        return None
    archive = RUNTIME_ROOT / f"SourceHanSansSC-{SOURCE_HAN_VERSION}.zip"
    print("[准备] 首次下载开源思源黑体（约 95 MB）...")
    _download(SOURCE_HAN_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_HAN_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeSetupError("思源黑体下载文件校验失败，已拒绝使用。")
    SOURCE_HAN_FONT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        member = next(
            (name for name in package.namelist() if name.endswith("SourceHanSansSC-Heavy.otf")),
            None,
        )
        if not member:
            raise RuntimeSetupError("思源黑体压缩包中找不到 Heavy 字重。")
        with package.open(member) as source, SOURCE_HAN_FONT.with_suffix(".tmp.otf").open("wb") as target:
            shutil.copyfileobj(source, target)
    os.replace(SOURCE_HAN_FONT.with_suffix(".tmp.otf"), SOURCE_HAN_FONT)
    archive.unlink(missing_ok=True)
    return SOURCE_HAN_FONT
