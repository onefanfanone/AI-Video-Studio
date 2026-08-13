from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT.parent / "AI-Video-Studio-Public"
DENIED_PARTS = {
    ".git",
    ".venv",
    ".runtime",
    ".cache",
    "outputs",
    "raw",
    "projects",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".bat",
    ".ps1",
    ".toml",
    ".css",
    ".js",
    ".example",
}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
    re.compile("(?i)C:" + r"\\Users\\" + "one" + "fan"),
    re.compile("(?i)E:" + r"\\" + "AI-Video-" + "Data"),
]


def allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in DENIED_PARTS or part.endswith(".pyc") for part in relative.parts):
        return False
    first = relative.parts[0]
    if first in {"app", "config", "release", "tests", "tools"}:
        return True
    return relative.as_posix() in {
        ".env.local.example",
        ".gitignore",
        "AGENTS.md",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "history_image_api.json",
        "requirements.txt",
        "run.bat",
    }


def scan_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"run.bat", ".gitignore"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            raise RuntimeError(f"公开导出被阻止：{path} 命中敏感模式 {pattern.pattern}")


def build(destination: Path) -> dict[str, object]:
    destination = destination.expanduser().resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise RuntimeError("公开导出目录不能位于私有源码目录内部。")
    if destination.exists():
        if (destination / ".git").exists():
            for child in destination.iterdir():
                if child.name == ".git":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for source in sorted(ROOT.rglob("*")):
        if not source.is_file() or not allowed(source):
            continue
        scan_text(source)
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        files.append({"path": relative.as_posix(), "bytes": target.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "files": files,
    }
    (destination / "PUBLIC_EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"destination": str(destination), "file_count": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(json.dumps(build(args.destination), ensure_ascii=False, indent=2))
