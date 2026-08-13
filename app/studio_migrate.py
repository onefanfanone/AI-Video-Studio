from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .studio_settings import (
    CODE_ROOT,
    SecretStore,
    SettingsStore,
    StudioSettingsError,
    ensure_workspace,
    import_legacy_env,
)


DATA_MAPPINGS = {
    ".runtime": "runtime",
    ".cache": "cache",
    "raw": "raw",
    "outputs": "outputs",
}


class MigrationError(RuntimeError):
    """A reversible workspace migration failure."""


@dataclass(frozen=True)
class FileRecord:
    relative: str
    size: int
    sha256: str


def _is_reparse_point(path: Path) -> bool:
    try:
        return bool(path.lstat().st_file_attributes & 0x400)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return path.is_symlink()


def _process_running(image_name: str) -> bool:
    if os.name != "nt":
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return image_name.lower() in completed.stdout.lower()


def locking_processes(path: Path) -> list[dict[str, Any]]:
    """Return Windows Restart Manager processes holding files below *path*."""
    if os.name != "nt" or not path.exists():
        return []

    class RM_UNIQUE_PROCESS(ctypes.Structure):
        _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]

    class RM_PROCESS_INFO(ctypes.Structure):
        _fields_ = [
            ("Process", RM_UNIQUE_PROCESS),
            ("strAppName", wintypes.WCHAR * 256),
            ("strServiceShortName", wintypes.WCHAR * 64),
            ("ApplicationType", wintypes.DWORD),
            ("AppStatus", wintypes.ULONG),
            ("TSSessionId", wintypes.DWORD),
            ("bRestartable", wintypes.BOOL),
        ]

    resources = [str(item.resolve()) for item in path.rglob("*") if item.is_file()]
    if path.is_file():
        resources = [str(path.resolve())]
    if not resources:
        return []
    manager = ctypes.WinDLL("rstrtmgr")
    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(64)
    if manager.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return []
    try:
        array_type = wintypes.LPCWSTR * len(resources)
        result = manager.RmRegisterResources(
            session,
            len(resources),
            array_type(*resources),
            0,
            None,
            0,
            None,
        )
        if result != 0:
            return []
        needed = wintypes.UINT(0)
        count = wintypes.UINT(0)
        reasons = wintypes.DWORD(0)
        result = manager.RmGetList(
            session,
            ctypes.byref(needed),
            ctypes.byref(count),
            None,
            ctypes.byref(reasons),
        )
        if result not in {0, 234} or not needed.value:
            return []
        entries = (RM_PROCESS_INFO * needed.value)()
        count.value = needed.value
        result = manager.RmGetList(
            session,
            ctypes.byref(needed),
            ctypes.byref(count),
            entries,
            ctypes.byref(reasons),
        )
        if result != 0:
            return []
        return [
            {
                "pid": int(entry.Process.dwProcessId),
                "name": str(entry.strAppName),
                "restartable": bool(entry.bRestartable),
            }
            for entry in entries[: count.value]
        ]
    finally:
        manager.RmEndSession(session)


def _validated_paths(source: Path, target: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if source != CODE_ROOT.resolve():
        raise MigrationError(f"迁移源必须是当前源码目录：{CODE_ROOT}")
    if source == target or source in target.parents or target in source.parents:
        raise MigrationError("迁移目标不能与源码目录相同，也不能互相嵌套。")
    if target.anchor == str(target) or len(target.parts) < 2:
        raise MigrationError("拒绝把磁盘根目录作为工作区。")
    return source, target


def _file_records(root: Path) -> list[FileRecord]:
    result: list[FileRecord] = []
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        result.append(
            FileRecord(path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest())
        )
    return result


def _verify_tree(source: Path, target: Path) -> dict[str, Any]:
    expected = _file_records(source)
    actual = _file_records(target)
    if expected != actual:
        expected_map = {row.relative: row for row in expected}
        actual_map = {row.relative: row for row in actual}
        missing = sorted(set(expected_map) - set(actual_map))[:8]
        changed = sorted(
            name
            for name in set(expected_map) & set(actual_map)
            if expected_map[name] != actual_map[name]
        )[:8]
        raise MigrationError(
            f"复制校验失败：{source.name}；缺失 {missing}；不一致 {changed}。"
        )
    return {
        "files": len(expected),
        "bytes": sum(item.size for item in expected),
    }


def _copy_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                destination.is_file()
                and destination.stat().st_size == path.stat().st_size
                and destination.stat().st_mtime_ns == path.stat().st_mtime_ns
            ):
                continue
            shutil.copy2(path, destination)


def _junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        raise MigrationError("junction 迁移只支持 Windows。")
    completed = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode or not link.is_dir() or not _is_reparse_point(link):
        raise MigrationError(f"无法创建 junction：{link} → {target}\n{completed.stdout}")


def _draft_media_check(source: Path) -> dict[str, Any]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    draft_root = local / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft"
    references: set[Path] = set()
    if draft_root.is_dir():
        source_text = str(source).lower().replace("\\", "/")
        for draft_file in draft_root.glob("*/draft_content.json"):
            try:
                payload = json.loads(draft_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            def walk(value: Any) -> None:
                if isinstance(value, dict):
                    for child in value.values():
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
                elif isinstance(value, str):
                    normalized = value.lower().replace("\\", "/")
                    if source_text in normalized and "/outputs/" in normalized:
                        references.add(Path(value))

            walk(payload)
    missing = sorted(str(path) for path in references if not path.is_file())
    if missing:
        raise MigrationError("迁移后剪映草稿仍有缺失媒体：" + "；".join(missing[:10]))
    return {"references": len(references), "missing": 0}


def migration_plan(source: Path, target: Path) -> dict[str, Any]:
    source, target = _validated_paths(source, target)
    items = []
    total = 0
    for source_name, target_name in DATA_MAPPINGS.items():
        src = source / source_name
        size = sum(path.stat().st_size for path in src.rglob("*") if path.is_file()) if src.is_dir() else 0
        total += size
        items.append(
            {
                "source": str(src),
                "target": str(target / target_name),
                "exists": src.is_dir(),
                "junction": True,
                "bytes": size,
            }
        )
    items.append(
        {
            "source": str(source / "projects"),
            "target": str(target / "projects"),
            "exists": (source / "projects").is_dir(),
            "junction": False,
            "bytes": sum(path.stat().st_size for path in (source / "projects").rglob("*") if path.is_file()),
        }
    )
    return {"source": str(source), "target": str(target), "total_bytes": total, "items": items}


def migrate_workspace(
    target: Path,
    *,
    source: Path = CODE_ROOT,
    apply: bool = False,
    import_secrets: bool = True,
) -> dict[str, Any]:
    source, target = _validated_paths(source, target)
    plan = migration_plan(source, target)
    if not apply:
        return {"status": "dry_run", **plan}
    if _process_running("JianyingPro.exe"):
        raise MigrationError("检测到剪映正在运行。请关闭剪映后再迁移，避免草稿媒体引用冲突。")
    locked: list[dict[str, Any]] = []
    for source_name in DATA_MAPPINGS:
        locked.extend(locking_processes(source / source_name))
    blocking_locks = [
        item
        for item in locked
        if item.get("restartable")
        or not any(
            marker in str(item.get("name") or "").lower()
            for marker in ("defender", "antimalware", "msmpeng")
        )
    ]
    if blocking_locks:
        unique = {(item["pid"], item["name"]) for item in blocking_locks}
        detail = "、".join(f"{name} (PID {pid})" for pid, name in sorted(unique))
        raise MigrationError(f"迁移目录仍被这些程序占用：{detail}。请关闭后重试。")
    target.mkdir(parents=True, exist_ok=True)
    ensure_workspace(target)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups: dict[Path, Path] = {}
    verifications: dict[str, Any] = {}
    preserved_legacy_directories: dict[str, str] = {}
    try:
        for source_name, target_name in DATA_MAPPINGS.items():
            src = source / source_name
            dst = target / target_name
            if not src.is_dir():
                continue
            if _is_reparse_point(src):
                continue
            _copy_tree(src, dst)
            verifications[source_name] = _verify_tree(src, dst)
        projects = source / "projects"
        if projects.is_dir():
            _copy_tree(projects, target / "projects")
            verifications["projects"] = _verify_tree(projects, target / "projects")
        workflow = source / "history_image_api.json"
        if workflow.is_file():
            shutil.copy2(workflow, target / "workflows" / workflow.name)
        for source_name, target_name in DATA_MAPPINGS.items():
            src = source / source_name
            if not src.is_dir() or _is_reparse_point(src):
                continue
            backup = source / f"{source_name}.pre-studio-backup-{timestamp}"
            if backup.exists():
                raise MigrationError(f"回滚备份已存在：{backup}")
            try:
                os.replace(src, backup)
            except PermissionError as exc:
                if source_name == "outputs":
                    # Some media applications keep a directory handle open without
                    # reporting the individual file through Restart Manager. Keeping
                    # the old, already-verified output tree is safer than breaking an
                    # existing Jianying absolute media reference. New builds still use
                    # the migrated workspace selected below.
                    preserved_legacy_directories[str(src)] = (
                        "Windows 拒绝重命名旧 outputs；已保留原目录以兼容既有剪映草稿。"
                    )
                    continue
                raise MigrationError(
                    f"Windows 拒绝重命名 {src}。请关闭正在使用该目录的程序后重试。"
                ) from exc
            backups[src] = backup
            try:
                _junction(src, target / target_name)
            except Exception:
                if src.exists():
                    os.rmdir(src)
                os.replace(backup, src)
                backups.pop(src, None)
                raise
        draft_check = _draft_media_check(source)
        settings = SettingsStore().initialize(target)
        imported: list[str] = []
        if import_secrets and (source / ".env.local").is_file():
            imported = import_legacy_env(source / ".env.local", SecretStore())
        report = {
            "status": "migrated",
            "source": str(source),
            "target": str(target),
            "verifications": verifications,
            "draft_check": draft_check,
            "backups": {str(key): str(value) for key, value in backups.items()},
            "preserved_legacy_directories": preserved_legacy_directories,
            "nonblocking_security_scanners": [
                {"pid": item["pid"], "name": item["name"]}
                for item in locked
                if item not in blocking_locks
            ],
            "secret_names_imported": imported,
            "legacy_env_deleted": False,
            "settings": {
                "workspace": settings["workspace"],
                "jianying_draft_root": settings["jianying_draft_root"],
            },
            "next_step": (
                "确认剪映草稿和任务续跑后，再人工删除 pre-studio-backup 目录；"
                "若报告保留了旧 outputs，请勿删除，除非先改为指向新工作区的 junction。"
            ),
        }
        report_path = SettingsStore().root / "state" / "migration-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    except Exception:
        # Roll back only the directory swaps made during this invocation. Copied target
        # data remains harmless and can make a later retry resumable.
        for src, backup in reversed(list(backups.items())):
            try:
                if src.exists() and _is_reparse_point(src):
                    os.rmdir(src)
                if backup.exists() and not src.exists():
                    os.replace(backup, src)
            except OSError:
                pass
        raise
