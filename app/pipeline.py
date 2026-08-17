from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image, ImageOps

from .jianying_text import add_rich_subtitles
from .asset_review import prepare_and_deduplicate, run_review_server
from .asset_reuse import (
    AssetReuseError,
    build_reuse_plan,
    merge_reused_candidates,
    run_reuse_review_server,
    unmatched_scene_plan,
)
from .comfyui_client import ComfyUIError, resolve_workflow_path, validate_server_url
from .openai_visuals import load_local_env
from .visual_planner import VisualPlannerError, create_visual_plan_with_audit
from .visual_semantics import SemanticReviewError, review_asset_candidates
from .visual_supply import (
    VisualSupplyError,
    add_ai_fallbacks,
    apply_review_request,
    build_ai_only_candidates,
    build_search_results,
    build_sourced_storyboard,
    download_selected_assets,
    validate_asset_manifest,
    write_license_outputs,
)
from .mpt_runtime import (
    MPT_COMMIT,
    MPT_VERSION,
    RUNTIME_ROOT,
    RuntimeSetupError,
    ensure_source_han_font,
    invoke_worker,
)
from .task_state import REUSE_TASK_STAGES, TASK_STAGES, TaskState, find_task_dir
from .transcript import (
    AlignmentQualityError,
    Cue,
    build_transcript,
    cues_from_transcript,
    wrap_caption as wrap_transcript_caption,
)
from .studio_settings import discover_jianying_draft_root, get_studio_paths
from . import mpt_runtime as _mpt_runtime


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
SRT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)
HARD_END = "。！？!?；;"
SOFT_END = "，,：:"
_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


class BuildError(RuntimeError):
    """A user-facing build failure."""


def _source_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _SOURCE_DIGEST_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SOURCE_DIGEST_CACHE[key] = value
    return value


def configure_pipeline_paths() -> None:
    """Refresh runtime globals after onboarding or workspace migration."""
    global RUNTIME_ROOT
    RUNTIME_ROOT = get_studio_paths(ROOT).runtime_root
    _mpt_runtime.configure_runtime_root(RUNTIME_ROOT)

@dataclass(frozen=True)
class MediaInfo:
    path: Path
    kind: str
    width: int
    height: int
    duration: float
    fps: float
    codec_name: str
    pix_fmt: str
    color_range: str
    audio_codec: str


def _run(
    args: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in args]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise BuildError(f"命令执行失败：{' '.join(command[:3])}\n{detail[-3000:]}")
    return completed


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolve_project(project: str) -> Path:
    candidate = Path(project)
    if not candidate.is_absolute():
        direct = (Path.cwd() / candidate).resolve()
        paths = get_studio_paths(ROOT)
        named = (paths.project_root / project).resolve()
        legacy = (ROOT / "projects" / project).resolve()
        candidate = direct if direct.exists() else named if named.exists() else legacy
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise BuildError(f"找不到项目目录：{candidate}")
    return candidate


def _load_config(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "project.yaml"
    if not config_path.is_file():
        raise BuildError(f"缺少项目配置：{config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != 1:
        raise BuildError("project.yaml 必须是 version: 1 的配置。")
    return config


def _load_named_config(path: Path, key: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BuildError(f"无法读取配置：{path}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise BuildError(f"配置必须是 version: 1：{path}")
    values = payload.get(key)
    if not isinstance(values, dict):
        raise BuildError(f"配置缺少 {key}：{path}")
    return values


def resolve_voice_config(voice_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a named stage-two profile while keeping legacy YAML compatible."""
    if voice_config.get("voice"):
        return {
            **voice_config,
            "provider": voice_config.get("provider", "moneyprinter_edge"),
            "profile": voice_config.get("profile", "snapshot"),
            "label": voice_config.get("label", voice_config["voice"]),
            "rate": voice_config.get("rate", "+0%"),
            "pitch": voice_config.get("pitch", "+0Hz"),
            "pronunciation": voice_config.get("pronunciation", {}),
            "pauses": voice_config.get("pauses", {}),
        }
    if "profile" not in voice_config:
        direct_voice = voice_config.get("voice") or voice_config.get("name")
        if not direct_voice:
            raise BuildError("voice 必须包含 profile、voice 或旧版 name。")
        return {
            **voice_config,
            "provider": voice_config.get("provider", "edge_direct"),
            "profile": "legacy",
            "voice": direct_voice,
            "label": voice_config.get("label", direct_voice),
            "rate": voice_config.get("rate", "+0%"),
            "pitch": voice_config.get("pitch", "+0Hz"),
            "pronunciation": voice_config.get("pronunciation", {}),
            "pauses": voice_config.get("pauses", {}),
        }
    profiles = _load_named_config(ROOT / "config" / "voice_profiles.yaml", "profiles")
    profile_id = str(voice_config["profile"])
    if profile_id not in profiles:
        raise BuildError(f"未知配音 profile：{profile_id}")
    profile = profiles[profile_id]
    if not isinstance(profile, dict):
        raise BuildError(f"配音 profile 格式错误：{profile_id}")
    return {
        **profile,
        **voice_config,
        "profile": profile_id,
        "provider": voice_config.get("provider", "moneyprinter_edge"),
        "voice": profile["voice"],
        "rate": voice_config.get("rate", profile.get("rate", "+0%")),
        "pitch": voice_config.get("pitch", profile.get("pitch", "+0Hz")),
        "pronunciation": voice_config.get("pronunciation", {}),
        "pauses": voice_config.get("pauses", {}),
    }


def resolve_subtitle_config(subtitle_config: dict[str, Any]) -> dict[str, Any]:
    preset_id = str(subtitle_config.get("style_preset", "history_clean"))
    presets = _load_named_config(ROOT / "config" / "subtitle_presets.yaml", "presets")
    if preset_id not in presets:
        raise BuildError(f"未知字幕预设：{preset_id}")
    return {**presets[preset_id], **subtitle_config, "style_preset": preset_id}


def _fraction_to_float(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_media(path: Path) -> MediaInfo:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,color_range,width,height,r_frame_rate,duration:format=duration",
            "-of",
            "json",
            path,
        ]
    )
    payload = json.loads(completed.stdout)
    video = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if not video:
        raise BuildError(f"无法读取视频画面信息：{path.name}")
    audio = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    duration_text = payload.get("format", {}).get("duration") or video.get("duration") or 0
    try:
        duration = float(duration_text)
    except (TypeError, ValueError):
        duration = 0.0
    kind = "image" if path.suffix.lower() in IMAGE_EXTENSIONS else "video"
    return MediaInfo(
        path=path,
        kind=kind,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        duration=duration,
        fps=_fraction_to_float(video.get("r_frame_rate")),
        codec_name=str(video.get("codec_name") or ""),
        pix_fmt=str(video.get("pix_fmt") or ""),
        color_range=str(video.get("color_range") or ""),
        audio_codec=str(audio.get("codec_name") or "") if audio else "",
    )


def probe_audio_duration(path: Path) -> float:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise BuildError(f"无法读取音频时长：{path}") from exc


def _jianying_running() -> bool:
    if sys.platform != "win32":
        return False
    completed = _run(
        ["tasklist", "/FI", "IMAGENAME eq JianyingPro.exe", "/FO", "CSV", "/NH"],
        check=False,
    )
    return "jianyingpro.exe" in completed.stdout.lower()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _preflight(
    project_dir: Path,
    config: dict[str, Any],
    draft_root_override: str | None,
    skip_draft: bool,
    visual_mode: str,
) -> tuple[Path, Path, Path]:
    for command in ("ffmpeg", "ffprobe"):
        if not shutil.which(command):
            raise BuildError(f"找不到 {command}，请安装后加入 PATH。")

    project_config = config["project"]
    script_path = (project_dir / project_config["script_file"]).resolve()
    raw_dir = (project_dir / config["media"]["raw_dir"]).resolve()
    if not script_path.is_file():
        raise BuildError(f"找不到文案文件：{script_path}")
    if visual_mode == "local":
        if not raw_dir.is_dir():
            raise BuildError(f"找不到素材目录：{raw_dir}")
        has_media = any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            for item in raw_dir.iterdir()
        )
        if not has_media:
            raise BuildError(f"素材目录为空：{raw_dir}")
    else:
        visuals_config = config.get("visuals", {})
        visual_strategy = str(
            visuals_config.get("strategy", "museum_and_ai")
        ).lower()
        if visual_strategy not in {"museum_and_ai", "ai_only"}:
            raise BuildError(
                "visuals.strategy 只能是 museum_and_ai 或 ai_only。"
            )
        planner = visuals_config.get("planner", {})
        planner_provider = str(planner.get("provider", "openai")).lower()
        env = load_local_env(ROOT)
        secret_ref = str(
            planner.get("secret_ref")
            or ("DEEPSEEK_API_KEY" if planner_provider in {"deepseek", "chat_completions"} else "OPENAI_API_KEY")
        )
        if planner_provider in {"deepseek", "chat_completions"} and not env.get(secret_ref):
            raise BuildError(
                "sourced 模式的 DeepSeek 场景规划需要 DEEPSEEK_API_KEY。"
                "请复制 .env.local.example 为 .env.local 并填入密钥。"
            )
        if planner_provider == "openai" and not env.get(secret_ref):
            raise BuildError(
                "sourced 模式的 OpenAI 场景规划需要 OPENAI_API_KEY。"
                "也可以把 planner.provider 改为 deepseek。"
            )
        if planner_provider not in {"deepseek", "chat_completions", "openai"}:
            raise BuildError(f"不支持的场景规划 provider：{planner_provider}")
        ai_config = visuals_config.get("ai_fallback", {})
        ai_provider = str(ai_config.get("provider", "openai")).lower()
        if ai_config.get("enabled", True) and ai_provider == "comfyui_local":
            try:
                validate_server_url(ai_config)
                resolve_workflow_path(ai_config, project_dir)
            except ComfyUIError as exc:
                raise BuildError(str(exc)) from exc
        elif ai_config.get("enabled", True) and ai_provider != "openai":
            raise BuildError(f"不支持的 AI 生图 provider：{ai_provider}")

    configured_root = Path(
        str(config.get("jianying", {}).get("draft_root") or discover_jianying_draft_root())
    ).expanduser().resolve()
    if draft_root_override:
        override = Path(draft_root_override)
        draft_root = (ROOT / override).resolve() if not override.is_absolute() else override.resolve()
    else:
        draft_root = configured_root

    if not skip_draft and _same_path(draft_root, configured_root) and not draft_root.is_dir():
        raise BuildError(f"剪映草稿目录不存在：{draft_root}")
    return script_path, raw_dir, draft_root


def _timestamp_to_seconds(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _seconds_to_srt(value: float) -> str:
    milliseconds = max(0, int(round(value * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _seconds_to_ass(value: float) -> str:
    centiseconds = max(0, int(round(value * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def parse_srt(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = SRT_TIME_RE.search(lines[timing_index])
        if not match:
            continue
        cue_text = "\n".join(lines[timing_index + 1 :]).strip()
        if cue_text:
            cues.append(
                Cue(
                    _timestamp_to_seconds(match.group("start")),
                    _timestamp_to_seconds(match.group("end")),
                    cue_text,
                )
            )
    if not cues:
        raise BuildError(f"没有从字幕文件中读取到有效时间轴：{path}")
    return cues


def parse_edge_metadata(path: Path) -> list[Cue]:
    """Convert edge-tts 7.x JSONL boundary metadata to timestamped cues."""
    cues: list[Cue] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildError(f"Edge TTS 元数据第 {line_number} 行不是有效 JSON。") from exc
        if item.get("type") not in {"SentenceBoundary", "WordBoundary"}:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        offset = float(item.get("offset", 0)) / 10_000_000
        duration = float(item.get("duration", 0)) / 10_000_000
        if duration > 0:
            cues.append(Cue(offset, offset + duration, text))
    if not cues:
        raise BuildError(f"没有从 Edge TTS 元数据中读取到有效时间轴：{path}")
    return cues


def edge_metadata_alignment(path: Path) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildError(f"Edge TTS 元数据第 {line_number} 行不是有效 JSON。") from exc
        if item.get("type") not in {"WordBoundary", "SentenceBoundary"}:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("offset", 0)) / 10_000_000
        duration = float(item.get("duration", 0)) / 10_000_000
        if duration <= 0:
            continue
        words.append(
            {
                "text": text,
                "start": start,
                "end": start + duration,
                "probability": 1.0,
            }
        )
    if not words:
        raise BuildError(f"Edge TTS 元数据没有可用的时间边界：{path}")
    return {
        "schema_version": 1,
        "engine": "edge-boundary-explicit",
        "model": None,
        "device": None,
        "words": words,
    }


def _split_long_cue(cue: Cue, max_chars: int) -> list[Cue]:
    text = re.sub(r"\s+", "", cue.text)
    if len(text) <= max_chars:
        return [Cue(cue.start, cue.end, text)]
    chunks = [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    total_chars = sum(len(chunk) for chunk in chunks)
    cursor = cue.start
    result: list[Cue] = []
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            end = cue.end
        else:
            end = cursor + (cue.end - cue.start) * len(chunk) / total_chars
        result.append(Cue(cursor, end, chunk))
        cursor = end
    return result


def wrap_caption(text: str, max_chars_per_line: int, max_lines: int) -> str:
    return wrap_transcript_caption(text, max_chars_per_line, max_lines)


def merge_cues(
    cues: Sequence[Cue],
    max_chars_per_line: int,
    max_lines: int,
) -> list[Cue]:
    capacity = max_chars_per_line * max_lines
    expanded = [part for cue in cues for part in _split_long_cue(cue, capacity)]
    merged: list[Cue] = []
    current: Cue | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            merged.append(
                Cue(
                    current.start,
                    current.end,
                    wrap_caption(current.text, max_chars_per_line, max_lines),
                )
            )
        current = None

    for cue in expanded:
        clean = re.sub(r"\s+", "", cue.text)
        if not clean:
            continue
        if current is None:
            current = Cue(cue.start, cue.end, clean)
        else:
            combined = current.text.replace("\n", "") + clean
            gap = cue.start - current.end
            previous_ended = current.text.rstrip().endswith(tuple(HARD_END))
            if len(combined) <= capacity and gap <= 0.45 and not previous_ended:
                current = Cue(current.start, cue.end, combined)
            else:
                flush()
                current = Cue(cue.start, cue.end, clean)

        flat = current.text.replace("\n", "")
        if flat.endswith(tuple(HARD_END)) or (
            flat.endswith(tuple(SOFT_END)) and len(flat) >= max_chars_per_line
        ):
            flush()
    flush()

    normalized: list[Cue] = []
    for cue in merged:
        start = max(cue.start, normalized[-1].end if normalized else cue.start)
        if cue.end > start:
            normalized.append(Cue(start, cue.end, cue.text))
    return normalized


def write_srt(path: Path, cues: Sequence[Cue]) -> None:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{_seconds_to_srt(cue.start)} --> {_seconds_to_srt(cue.end)}\n{cue.text}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_ass(
    path: Path,
    cues: Sequence[Cue],
    canvas: dict[str, Any],
    subtitle_config: dict[str, Any],
    transcript: dict[str, Any] | None = None,
    disclosure: dict[str, Any] | None = None,
) -> None:
    def ass_color(value: str, opacity: float = 1.0) -> str:
        value = value.lstrip("#")
        if len(value) != 6:
            value = "FFFFFF"
        alpha = round(255 * (1.0 - min(1.0, max(0.0, opacity))))
        return f"&H{alpha:02X}{value[4:6]}{value[2:4]}{value[0:2]}"

    primary = ass_color(str(subtitle_config.get("base_color", "#FFFFFF")))
    highlight = ass_color(str(subtitle_config.get("highlight_color", "#FFD54A")))
    outline_color = ass_color(str(subtitle_config.get("outline_color", "#101010")))
    shadow_color = ass_color(
        str(subtitle_config.get("shadow_color", "#000000")),
        float(subtitle_config.get("shadow_opacity", 0.5)),
    )
    bold = -1 if bool(subtitle_config.get("bold", True)) else 0
    letter_spacing = float(subtitle_config.get("letter_spacing", 0))
    fade_in_ms = max(0, int(subtitle_config.get("fade_in_ms", 0)))
    fade_out_ms = max(0, int(subtitle_config.get("fade_out_ms", 0)))
    fade_tag = rf"{{\fad({fade_in_ms},{fade_out_ms})}}" if fade_in_ms or fade_out_ms else ""
    highlight_inline = highlight.replace("&H00", "&H", 1) + "&"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {canvas['width']}
PlayResY: {canvas['height']}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{subtitle_config['font_name']},{subtitle_config['font_size']},{primary},{primary},{outline_color},{shadow_color},{bold},0,0,0,100,100,{letter_spacing},0,1,{subtitle_config['outline']},{subtitle_config['shadow']},2,70,70,{subtitle_config['margin_bottom']},1
Style: Disclosure,{subtitle_config['font_name']},{max(38, int(subtitle_config['font_size']) - 10)},&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,70,70,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    transcript_cues = transcript.get("cues", []) if transcript else []
    for index, cue in enumerate(cues):
        text = cue.text.replace("{", "（").replace("}", "）").replace("\n", r"\N")
        text = fade_tag + text
        if index == 0 and subtitle_config.get("style_preset") == "history_hook":
            text = r"{\fscx94\fscy94\t(0,140,\fscx100\fscy100)\fad(80,0)}" + text
        events.append(
            f"Dialogue: 0,{_seconds_to_ass(cue.start)},{_seconds_to_ass(cue.end)},Default,,0,0,0,,{text}"
        )
        if index >= len(transcript_cues):
            continue
        emphasis = transcript_cues[index].get("emphasis")
        if not emphasis or subtitle_config.get("style_preset") == "history_clean":
            continue
        original = cue.text.replace("{", "（").replace("}", "）")
        start = int(emphasis["text_start"])
        end = int(emphasis["text_end"])
        visible_count = 0
        actual_start = 0
        actual_end = len(original)
        for position, character in enumerate(original):
            if character == "\n":
                continue
            if visible_count == start:
                actual_start = position
            if visible_count == end:
                actual_end = position
                break
            visible_count += 1
        prefix = original[:actual_start].replace("\n", r"\N")
        keyword = original[actual_start:actual_end].replace("\n", r"\N")
        suffix = original[actual_end:].replace("\n", r"\N")
        overlay = (
            r"{\alpha&HFF&}" + prefix
            + r"{\alpha&H00&\1c" + highlight_inline + "}" + keyword
            + r"{\alpha&HFF&}" + suffix
        )
        events.append(
            "Dialogue: 1,"
            f"{_seconds_to_ass(float(emphasis['start']))},"
            f"{_seconds_to_ass(float(emphasis['end']))},"
            f"Default,,0,0,0,,{overlay}"
        )
    if disclosure and disclosure.get("required"):
        end = float(disclosure["end"])
        start = max(0.0, end - float(disclosure.get("seconds", 2.0)))
        text = str(disclosure.get("text", "部分画面为 AI 历史重构")).replace("{", "（").replace("}", "）")
        events.append(
            f"Dialogue: 3,{_seconds_to_ass(start)},{_seconds_to_ass(end)},Disclosure,,0,0,0,,{text}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


async def _synthesize_tts(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    audio_path: Path,
    metadata_path: Path,
) -> None:
    import edge_tts

    communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicator.save(str(audio_path), str(metadata_path))


def _prepare_spoken_text(text: str, voice_config: dict[str, Any]) -> str:
    spoken = text
    pronunciation = voice_config.get("pronunciation", {})
    if pronunciation and not isinstance(pronunciation, dict):
        raise BuildError("voice.pronunciation 必须是“原文: 读音”的映射。")
    for source, replacement in sorted(
        (pronunciation or {}).items(), key=lambda item: len(str(item[0])), reverse=True
    ):
        spoken = spoken.replace(str(source), str(replacement))
    pauses = voice_config.get("pauses", {})
    if pauses and not isinstance(pauses, dict):
        raise BuildError("voice.pauses 必须是“短语: 毫秒”的映射。")
    for phrase, milliseconds in (pauses or {}).items():
        try:
            duration = int(milliseconds)
        except (TypeError, ValueError) as exc:
            raise BuildError(f"停顿时长必须是毫秒整数：{phrase}") from exc
        punctuation = "。" if duration >= 450 else "，"
        spoken = spoken.replace(str(phrase), str(phrase) + punctuation)
    return spoken


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_narration(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp.mp3")
    temporary.unlink(missing_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source,
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=7",
            "-ar",
            "48000",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            temporary,
        ]
    )
    os.replace(temporary, destination)


def create_or_reuse_tts(
    text: str,
    voice_config: dict[str, Any],
    run_dir: Path,
) -> tuple[Path, Path, Path, bool, dict[str, Any]]:
    resolved = resolve_voice_config(voice_config)
    spoken_text = _prepare_spoken_text(text, resolved)
    cache_root = get_studio_paths(ROOT).cache_root / "tts"
    cache_root.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "text": spoken_text,
        "provider": resolved["provider"],
        "profile": resolved["profile"],
        "voice": resolved["voice"],
        "rate": resolved["rate"],
        "pitch": resolved.get("pitch", "+0Hz"),
        "pronunciation": resolved.get("pronunciation", {}),
        "pauses": resolved.get("pauses", {}),
        "moneyprinterturbo": MPT_COMMIT if resolved["provider"] == "moneyprinter_edge" else None,
        "edge_tts": "7.2.8",
        "normalization": "loudnorm-I-16-TP-1.5-LRA-7",
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cached_raw_audio = cache_root / f"{cache_key}.raw.mp3"
    cached_audio = cache_root / f"{cache_key}.normalized.mp3"
    cached_metadata = cache_root / f"{cache_key}.jsonl"
    cache_hit = cached_raw_audio.is_file() and cached_audio.is_file() and cached_metadata.is_file()

    if not cache_hit:
        temporary_audio = cache_root / f"{cache_key}.tmp.raw.mp3"
        temporary_metadata = cache_root / f"{cache_key}.tmp.jsonl"
        temporary_audio.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        try:
            if resolved["provider"] == "moneyprinter_edge":
                response_path = cache_root / f"{cache_key}.worker-response.json"
                invoke_worker(
                    {
                        "operation": "tts",
                        "text": spoken_text,
                        "voice": resolved["voice"],
                        "rate": resolved["rate"],
                        "pitch": resolved.get("pitch", "+0Hz"),
                        "output_audio": str(temporary_audio),
                        "output_metadata": str(temporary_metadata),
                    },
                    response_path,
                )
            elif resolved["provider"] == "edge_direct":
                asyncio.run(
                    _synthesize_tts(
                        spoken_text,
                        resolved["voice"],
                        resolved["rate"],
                        resolved.get("pitch", "+0Hz"),
                        temporary_audio,
                        temporary_metadata,
                    )
                )
            else:
                raise BuildError(f"当前不支持配音提供商：{resolved['provider']}")
        except Exception as exc:
            temporary_audio.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            raise BuildError(
                "配音失败。首次生成需要可访问 MoneyPrinterTurbo 和微软语音服务的网络；"
                f"原始错误：{exc}"
            ) from exc
        if not temporary_audio.is_file() or not temporary_metadata.is_file():
            raise BuildError("Edge TTS 未生成完整的音频和时间元数据。")
        temporary_audio.replace(cached_raw_audio)
        temporary_metadata.replace(cached_metadata)
        _normalize_narration(cached_raw_audio, cached_audio)

    narration_raw = run_dir / "narration.raw.mp3"
    narration = run_dir / "narration.mp3"
    metadata = run_dir / "captions_metadata.jsonl"
    shutil.copy2(cached_raw_audio, narration_raw)
    shutil.copy2(cached_audio, narration)
    shutil.copy2(cached_metadata, metadata)
    return narration_raw, narration, metadata, cache_hit, resolved


def create_or_reuse_alignment(
    narration: Path,
    subtitle_config: dict[str, Any],
    run_dir: Path,
    *,
    initial_prompt: str = "",
) -> tuple[dict[str, Any], Path, bool]:
    cache_root = get_studio_paths(ROOT).cache_root / "alignment"
    cache_root.mkdir(parents=True, exist_ok=True)
    model = str(subtitle_config.get("whisper_model", "large-v3"))
    local_model = RUNTIME_ROOT / "models" / f"whisper-{model}"
    model_bin = local_model / "model.bin"
    device = str(subtitle_config.get("whisper_device", "cuda"))
    compute_type = str(
        subtitle_config.get("whisper_compute_type", "float16" if device == "cuda" else "int8")
    )
    key_payload = {
        "audio": _sha256_file(narration),
        "provider": subtitle_config.get("alignment_provider"),
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "moneyprinterturbo": MPT_COMMIT,
        "prompt_version": "whisper-hints-v1",
        "initial_prompt": initial_prompt,
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    cached = cache_root / f"{cache_key}.json"
    cache_hit = cached.is_file()
    if not cache_hit:
        auto_download = bool(subtitle_config.get("whisper_auto_download", False))
        if model_bin.is_file():
            worker_model = str(local_model)
        elif auto_download:
            worker_model = model
        else:
            raise BuildError(
                f"尚未安装本地 Whisper {model}，已按配置禁止自动下载。\n"
                f"官方下载页：https://huggingface.co/Systran/faster-whisper-{model}/tree/main\n"
                f"请把完整模型放到：{local_model}\n"
                f"完成标志文件应为：{model_bin}\n"
                "放好后再次双击 run.bat，即可自动从 alignment 阶段继续。"
            )
        temporary = cache_root / f"{cache_key}.tmp.json"
        response = cache_root / f"{cache_key}.worker-response.json"
        try:
            invoke_worker(
                {
                    "operation": "align",
                    "audio": str(narration),
                    "model": worker_model,
                    "device": device,
                    "compute_type": compute_type,
                    "download_root": str(RUNTIME_ROOT / "models"),
                    "output_alignment": str(temporary),
                    "initial_prompt": initial_prompt,
                },
                response,
            )
        except (RuntimeSetupError, Exception) as exc:
            temporary.unlink(missing_ok=True)
            raise BuildError(
                "Whisper 对齐失败。请检查本地模型、CUDA/cuDNN 运行库、"
                f"显卡驱动和磁盘空间。原始错误：{exc}"
            ) from exc
        if not temporary.is_file():
            raise BuildError("Whisper worker 未生成 alignment.json。")
        os.replace(temporary, cached)
    destination = run_dir / "working" / "alignment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    return payload, destination, cache_hit


def whisper_initial_prompt(
    script: str,
    voice_config: dict[str, Any],
    subtitle_config: dict[str, Any],
    *,
    maximum_chars: int = 200,
) -> str:
    """Build a bounded rare-term hint list without sending the full script."""
    candidates = [
        *[str(item) for item in (voice_config.get("pronunciation") or {}).keys()],
        *[
            str(item)
            for item in (
                subtitle_config.get("emphasis", {}).get("proper_nouns", [])
            )
        ],
    ]
    selected: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = candidate.strip()
        if not term or term not in script or term in seen:
            continue
        proposed = "、".join([*selected, term])
        if len(proposed) > maximum_chars:
            continue
        selected.append(term)
        seen.add(term)
    return "、".join(selected)


def schedule_shots(
    duration: float,
    fps: int,
    hook_until: float,
    hook_range: tuple[float, float],
    body_range: tuple[float, float],
    seed: int,
) -> list[dict[str, float]]:
    total_frames = max(1, round(duration * fps))
    hook_frames = round(hook_until * fps)
    randomizer = random.Random(seed)
    cursor = 0
    frame_ranges = (
        (max(1, round(hook_range[0] * fps)), max(1, round(hook_range[1] * fps))),
        (max(1, round(body_range[0] * fps)), max(1, round(body_range[1] * fps))),
    )
    shot_frames: list[int] = []
    while cursor < total_frames:
        bounds = frame_ranges[0] if cursor < hook_frames else frame_ranges[1]
        chosen = randomizer.randint(*bounds)
        remaining = total_frames - cursor
        minimum = frame_ranges[0][0] if cursor < hook_frames else frame_ranges[1][0]
        if remaining < chosen:
            chosen = remaining
        if remaining - chosen and remaining - chosen < minimum:
            chosen = remaining
        shot_frames.append(chosen)
        cursor += chosen

    shots: list[dict[str, float]] = []
    cursor = 0
    for frames in shot_frames:
        shots.append(
            {
                "start": cursor / fps,
                "duration": frames / fps,
                "end": (cursor + frames) / fps,
                "frames": frames,
            }
        )
        cursor += frames
    return shots


def _list_media(raw_dir: Path, media_config: dict[str, Any]) -> tuple[list[MediaInfo], list[str]]:
    excluded_names = {name.casefold() for name in media_config.get("exclude", [])}
    selected: list[MediaInfo] = []
    excluded: list[str] = []
    for path in sorted(raw_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        if path.name.casefold() in excluded_names:
            excluded.append(path.name)
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        selected.append(probe_media(path))
    if not any(item.kind == "image" for item in selected):
        raise BuildError("没有可用图片；本地素材模式至少需要一张图片素材。")
    return selected, excluded


def _caption_for_shot(shot: dict[str, float], cues: Sequence[Cue]) -> str:
    text = "".join(
        cue.text.replace("\n", "")
        for cue in cues
        if cue.end > shot["start"] and cue.start < shot["end"]
    )
    return text


def build_storyboard(
    title: str,
    duration: float,
    canvas: dict[str, Any],
    pacing: dict[str, Any],
    media_config: dict[str, Any],
    media: Sequence[MediaInfo],
    cues: Sequence[Cue],
) -> dict[str, Any]:
    seed = int(media_config["seed"])
    shots = schedule_shots(
        duration=duration,
        fps=int(canvas["fps"]),
        hook_until=float(pacing["hook_until_seconds"]),
        hook_range=tuple(float(value) for value in pacing["hook_shot_seconds"]),
        body_range=tuple(float(value) for value in pacing["body_shot_seconds"]),
        seed=seed,
    )
    images = [item for item in media if item.kind == "image"]
    videos = [item for item in media if item.kind == "video"]
    randomizer = random.Random(seed)
    randomizer.shuffle(images)

    requested_video_shots = min(int(media_config.get("video_shots", 0)), len(shots), 2)
    video_slots: list[int] = []
    if videos and requested_video_shots:
        candidates = [max(3, round(len(shots) * 0.34)), max(4, round(len(shots) * 0.72))]
        for candidate in candidates[:requested_video_shots]:
            slot = min(len(shots) - 1, candidate)
            if slot not in video_slots:
                video_slots.append(slot)

    overrides = media_config.get("assets", {})
    offsets = media_config.get("video_source_offsets", {})
    motion_cycle = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
    image_index = 0
    video_index = 0
    storyboard_shots: list[dict[str, Any]] = []

    for index, shot in enumerate(shots):
        if index in video_slots:
            asset = videos[video_index % len(videos)]
            video_offsets = offsets.get(asset.path.name, [0.0]) or [0.0]
            requested_offset = float(video_offsets[video_index % len(video_offsets)])
            maximum_offset = max(0.0, asset.duration - shot["duration"] - 0.1)
            source_start = min(maximum_offset, max(0.0, requested_offset))
            video_index += 1
            motion = "native"
        else:
            asset = images[image_index % len(images)]
            image_index += 1
            source_start = 0.0
            motion = motion_cycle[(index + seed) % len(motion_cycle)]

        override = overrides.get(asset.path.name, {})
        orientation = "landscape" if asset.width >= asset.height else "portrait"
        default_fit = (
            media_config["default_landscape_fit"]
            if orientation == "landscape"
            else media_config["default_portrait_fit"]
        )
        fit = override.get("fit", default_fit)
        if fit not in {"crop", "blur"}:
            raise BuildError(f"素材 {asset.path.name} 的 fit 只能是 crop 或 blur。")

        storyboard_shots.append(
            {
                "id": index + 1,
                **shot,
                "source": str(asset.path),
                "source_name": asset.path.name,
                "kind": asset.kind,
                "source_start": round(source_start, 3),
                "source_width": asset.width,
                "source_height": asset.height,
                "fit": fit,
                "focal_x": float(override.get("focal_x", 0.5)),
                "focal_y": float(override.get("focal_y", 0.5)),
                "motion": override.get("motion", motion),
                "caption_text": _caption_for_shot(shot, cues),
                "rendered_clip": f"working/clips/shot-{index + 1:03d}.mp4",
                "intent_id": None,
                "asset_id": f"local-{hashlib.sha256(str(asset.path).encode('utf-8')).hexdigest()[:16]}",
                "visual_origin": "local",
                "provenance_ref": None,
                "rights_code": "unverified_local",
                "ai_generated": False,
                "reviewed": False,
                "asset_score": None,
            }
        )

    return {
        "schema_version": 2,
        "visual_mode": "local",
        "title": title,
        "canvas": canvas,
        "audio_duration": duration,
        "visual_duration": shots[-1]["end"],
        "seed": seed,
        "draft_path": None,
        "shots": storyboard_shots,
    }


def _prepare_image_proxies(storyboard: dict[str, Any], run_dir: Path) -> dict[str, Path]:
    proxy_dir = run_dir / "working" / "proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxies: dict[str, Path] = {}
    image_sources = {
        shot["source"] for shot in storyboard["shots"] if shot["kind"] == "image"
    }
    for index, source_text in enumerate(sorted(image_sources), 1):
        source = Path(source_text)
        proxy = proxy_dir / f"image-{index:03d}.jpg"
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                image.save(proxy, "JPEG", quality=92, optimize=True)
        except Exception as exc:
            raise BuildError(f"无法生成图片代理：{source.name}；{exc}") from exc
        proxies[source_text] = proxy
    return proxies


def _zoompan_filter(
    motion: str,
    frames: int,
    width: int,
    height: int,
    fps: int,
    *,
    zoom_amount: float = 0.035,
    easing: str = "cosine",
) -> str:
    denominator = max(1, frames - 1)
    if not 0.0 <= zoom_amount <= 0.2:
        raise BuildError("画面运动的 zoom_amount 必须在 0 到 0.2 之间。")
    if easing == "cosine":
        progress = f"(1-cos(PI*on/{denominator}))/2"
    elif easing == "linear":
        progress = f"on/{denominator}"
    else:
        raise BuildError("画面运动的 easing 只能是 cosine 或 linear。")
    peak_zoom = 1.0 + zoom_amount
    if motion == "zoom_out":
        zoom = f"1.0+{zoom_amount:.6f}*(1-({progress}))"
        x = "iw/2-(iw/zoom/2)"
    elif motion == "pan_left":
        zoom = f"{peak_zoom:.6f}"
        x = f"(iw-iw/zoom)*(1-({progress}))"
    elif motion == "pan_right":
        zoom = f"{peak_zoom:.6f}"
        x = f"(iw-iw/zoom)*({progress})"
    elif motion == "static":
        zoom = "1.0"
        x = "0"
    else:
        zoom = f"1.0+{zoom_amount:.6f}*({progress})"
        x = "iw/2-(iw/zoom/2)"
    y = "ih/2-(ih/zoom/2)"
    return (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':"
        f"d={frames}:s={width}x{height}:fps={fps}"
    )


def _base_filter(shot: dict[str, Any], width: int, height: int) -> str:
    focal_x = min(1.0, max(0.0, float(shot["focal_x"])))
    focal_y = min(1.0, max(0.0, float(shot["focal_y"])))
    if shot["fit"] == "blur":
        foreground_width = round(width * 0.92)
        foreground_height = round(height * 0.86)
        return (
            "[0:v]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(iw-ow)*{focal_x:.3f}:(ih-oh)*{focal_y:.3f},"
            "gblur=sigma=32,eq=brightness=-0.12[bg2];"
            f"[fg]scale={foreground_width}:{foreground_height}:"
            "force_original_aspect_ratio=decrease[fg2];"
            "[bg2][fg2]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
        )
    overscan_width = round(width * 1.125)
    overscan_height = round(height * 1.125)
    return (
        f"[0:v]scale={overscan_width}:{overscan_height}:force_original_aspect_ratio=increase,"
        f"crop={overscan_width}:{overscan_height}:(iw-ow)*{focal_x:.3f}:"
        f"(ih-oh)*{focal_y:.3f},setsar=1[base]"
    )


def render_shot(
    shot: dict[str, Any],
    run_dir: Path,
    canvas: dict[str, Any],
    proxies: dict[str, Path],
    cache_root: Path | None = None,
) -> Path:
    width = int(canvas["width"])
    height = int(canvas["height"])
    fps = int(canvas["fps"])
    duration = float(shot["duration"])
    frames = int(shot["frames"])
    output = run_dir / shot["rendered_clip"]
    output.parent.mkdir(parents=True, exist_ok=True)
    source = proxies.get(shot["source"], Path(shot["source"]))
    render_cache: Path | None = None
    if cache_root is not None:
        source_digest = _source_sha256(source)
        render_key = hashlib.sha256(
            json.dumps(
                {
                    "version": "shot-render-v1",
                    "source_sha256": source_digest,
                    "kind": shot["kind"],
                    "source_start": shot.get("source_start", 0.0),
                    "frames": frames,
                    "duration": duration,
                    "fit": shot["fit"],
                    "focal_x": shot["focal_x"],
                    "focal_y": shot["focal_y"],
                    "motion": shot["motion"],
                    "canvas": canvas,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        render_cache = cache_root / "clips" / f"{render_key}.mp4"
        if render_cache.is_file() and render_cache.stat().st_size:
            try:
                cached_info = probe_media(render_cache)
            except BuildError:
                render_cache.unlink(missing_ok=True)
            else:
                if (
                    cached_info.width == width
                    and cached_info.height == height
                    and abs(cached_info.duration - duration) <= max(0.12, 1 / fps)
                ):
                    shutil.copy2(render_cache, output)
                    return output
                render_cache.unlink(missing_ok=True)

    if shot["kind"] == "image":
        motion_config = canvas.get("motion", {})
        working_scale = int(motion_config.get("working_scale", 2))
        if working_scale not in {1, 2, 3}:
            raise BuildError("画面运动的 working_scale 只能是 1、2 或 3。")
        zoom_amount = float(motion_config.get("zoom_amount", 0.035))
        easing = str(motion_config.get("easing", "cosine"))
        input_args: list[str | Path] = [
            "-i",
            source,
        ]
        motion_input = "base"
        filters = _base_filter(shot, width, height)
        if working_scale > 1:
            filters += (
                f";[base]scale=iw*{working_scale}:ih*{working_scale}:"
                "flags=lanczos[motionbase]"
            )
            motion_input = "motionbase"
        filters += (
            f";[{motion_input}]"
            + _zoompan_filter(
                shot["motion"],
                frames,
                width,
                height,
                fps,
                zoom_amount=zoom_amount,
                easing=easing,
            )
            + f",trim=duration={duration:.6f},setpts=PTS-STARTPTS,"
            "scale=in_range=auto:out_range=tv,format=yuv420p[v]"
        )
    else:
        input_args = [
            "-ss",
            f"{shot['source_start']:.3f}",
            "-i",
            source,
            "-t",
            f"{duration:.6f}",
        ]
        filters = (
            _base_filter(shot, width, height)
            + f";[base]fps={fps},trim=duration={duration:.6f},"
            "setpts=PTS-STARTPTS,scale=in_range=auto:out_range=tv,format=yuv420p[v]"
        )

    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *input_args,
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output,
        ]
    )
    if render_cache is not None:
        render_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = render_cache.with_suffix(".tmp.mp4")
        shutil.copy2(output, temporary)
        os.replace(temporary, render_cache)
    return output
def _concat_clips(clips: Sequence[Path], run_dir: Path) -> Path:
    list_path = run_dir / "working" / "clips.ffconcat"
    lines = ["ffconcat version 1.0"]
    for clip in clips:
        escaped = clip.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = run_dir / "working" / "visuals.mp4"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output,
        ]
    )
    return output


def _render_preview(
    visuals: Path,
    narration: Path,
    ass_path: Path,
    run_dir: Path,
    output_name: str,
    fonts_dir: Path | None = None,
) -> Path:
    output = run_dir / output_name
    subtitle_filter = f"ass={ass_path.name}"
    if fonts_dir:
        relative_fonts = fonts_dir.relative_to(run_dir).as_posix()
        subtitle_filter += f":fontsdir={relative_fonts}"
    subtitle_filter += ",scale=in_range=auto:out_range=tv,format=yuv420p"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            visuals,
            "-i",
            narration,
            "-vf",
            subtitle_filter,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-shortest",
            "-movflags",
            "+faststart",
            output,
        ],
        cwd=run_dir,
    )
    return output


def _create_jianying_draft(
    draft_root: Path,
    draft_name: str,
    storyboard: dict[str, Any],
    run_dir: Path,
    narration: Path,
    transcript: dict[str, Any],
    subtitle_config: dict[str, Any],
    disclosure: dict[str, Any] | None = None,
) -> Path:
    import pyJianYingDraft as draft

    draft_root.mkdir(parents=True, exist_ok=True)
    draft_path = draft_root / draft_name
    try:
        folder = draft.DraftFolder(str(draft_root))
        script = folder.create_draft(
            draft_name,
            int(storyboard["canvas"]["width"]),
            int(storyboard["canvas"]["height"]),
            int(storyboard["canvas"]["fps"]),
            maintrack_adsorb=True,
            allow_replace=False,
        )
        video_track = script.append_track(draft.TrackSpec(draft.TrackType.video, "video"))
        audio_track = script.append_track(draft.TrackSpec(draft.TrackType.audio, "narration"))

        timeline_cursor = 0
        for shot in storyboard["shots"]:
            clip_path = run_dir / shot["rendered_clip"]
            duration_units = round(
                int(shot["frames"]) * draft.SEC / int(storyboard["canvas"]["fps"])
            )
            segment = draft.VideoSegment(
                str(clip_path),
                draft.Timerange(timeline_cursor, duration_units),
                volume=0.0,
            )
            script.add_segment(segment, video_track)
            timeline_cursor += duration_units

        audio_duration = probe_audio_duration(narration)
        script.add_segment(
            draft.AudioSegment(
                str(narration),
                draft.Timerange(0, round(audio_duration * draft.SEC)),
                volume=1.0,
            ),
            audio_track,
        )
        add_rich_subtitles(
            script, draft, transcript, subtitle_config, disclosure=disclosure
        )
        script.save()
    except Exception as exc:
        if draft_path.exists():
            shutil.rmtree(draft_path, ignore_errors=True)
        raise BuildError(f"剪映草稿生成失败：{exc}") from exc
    return draft_path


def _package_versions() -> dict[str, str]:
    completed = _run(
        [str(Path(sys.executable)), "-m", "pip", "freeze", "--all"],
        check=False,
    )
    packages: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            if name.lower() in {
                "edge-tts",
                "pyjianyingdraft",
                "pyyaml",
                "pymediainfo",
                "imageio",
                "pillow",
                "numpy",
            }:
                packages[name] = version
    return packages


def validate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    storyboard_path = run_dir / "storyboard.json"
    narration_path = run_dir / "narration.mp3"
    captions_path = run_dir / "captions.srt"
    transcript_path = run_dir / "transcript.json"
    preview_candidates = list(run_dir.glob("preview*.mp4"))
    if not storyboard_path.is_file() or not narration_path.is_file() or not captions_path.is_file():
        raise BuildError(f"输出目录缺少 storyboard、旁白或字幕：{run_dir}")
    if len(preview_candidates) != 1:
        raise BuildError(f"输出目录中应当只有一个 preview MP4：{run_dir}")

    storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
    preview = probe_media(preview_candidates[0])
    audio_duration = probe_audio_duration(narration_path)
    cues = parse_srt(captions_path)
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8"))
        if transcript_path.is_file()
        else None
    )
    visual_mode = str(storyboard.get("visual_mode", "local"))
    ai_disclosure_required = False
    max_chars = int(transcript.get("limits", {}).get("max_chars_per_line", 14)) if transcript else 14
    max_lines = int(transcript.get("limits", {}).get("max_lines", 2)) if transcript else 2
    max_cue_seconds = (
        float(transcript.get("limits", {}).get("max_cue_seconds", 2.8))
        if transcript
        else None
    )
    min_cue_seconds = (
        float(transcript.get("limits", {}).get("min_cue_seconds", 0.8))
        if transcript
        else None
    )
    checks = {
        "preview_is_1080x1920": preview.width == 1080 and preview.height == 1920,
        "preview_is_30fps": abs(preview.fps - 30.0) < 0.01,
        "preview_is_h264_aac_yuv420p": (
            preview.codec_name == "h264"
            and preview.audio_codec == "aac"
            and preview.pix_fmt == "yuv420p"
        ),
        "duration_matches_audio_within_0_2s": abs(preview.duration - audio_duration) <= 0.2,
        "first_five_seconds_have_at_least_three_shots": sum(
            1 for shot in storyboard["shots"] if shot["start"] < 5.0
        )
        >= 3,
        "uses_images": any(shot["kind"] == "image" for shot in storyboard["shots"]),
        "uses_video": (
            visual_mode == "sourced"
            or any(shot["kind"] == "video" for shot in storyboard["shots"])
        ),
        "excluded_better_call_saul": all(
            shot["source_name"].casefold() != "better call saul.jpg"
            for shot in storyboard["shots"]
        ),
        "subtitles_are_monotonic": all(a.end <= b.start for a, b in zip(cues, cues[1:])),
        "subtitle_lines_are_within_configured_limit": all(
            len(cue.text.splitlines()) <= max_lines
            and all(len(line) <= max_chars for line in cue.text.splitlines())
            for cue in cues
        ),
        "storyboard_is_schema_v2": storyboard.get("schema_version") == 2,
        "storyboard_has_provenance_fields": all(
            all(
                key in shot
                for key in (
                    "intent_id", "asset_id", "visual_origin", "provenance_ref",
                    "rights_code", "ai_generated", "reviewed", "asset_score",
                )
            )
            for shot in storyboard["shots"]
        ),
    }
    if visual_mode == "sourced":
        manifest_path = run_dir / "assets_manifest.json"
        audit_path = run_dir / "license_audit.json"
        checks["sourced_assets_manifest_exists"] = manifest_path.is_file()
        checks["sourced_license_outputs_exist"] = all(
            (run_dir / name).is_file()
            for name in ("licenses.csv", "CREDITS.md", "AI_DISCLOSURE.md")
        ) and audit_path.is_file()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            ai_disclosure_required = bool(audit.get("ai_disclosure_required"))
            manifest_assets = {item["asset_id"]: item for item in manifest.get("assets", [])}
            checks["every_shot_is_human_reviewed_and_traceable"] = all(
                shot.get("reviewed")
                and shot.get("asset_id") in manifest_assets
                and (run_dir / str(shot.get("provenance_ref"))).is_file()
                for shot in storyboard["shots"]
            )
            checks["asset_hashes_match_manifest"] = all(
                hashlib.sha256(Path(item["local_path"]).read_bytes()).hexdigest()
                == item["sha256"]
                for item in manifest_assets.values()
            )
            checks["license_audit_ready"] = bool(
                audit.get("asset_rights_ready") and audit.get("human_reviewed")
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            checks["sourced_provenance_is_readable"] = False
    if transcript:
        checks["alignment_coverage_meets_threshold"] = float(
            transcript.get("alignment", {}).get("coverage", 0.0)
        ) >= float(transcript.get("alignment", {}).get("minimum_coverage", 0.98))
        checks["subtitle_cues_are_not_too_long"] = all(
            cue.end - cue.start <= float(max_cue_seconds) + 0.02 for cue in cues
        )
        checks["subtitle_cues_are_not_too_short"] = all(
            cue.end - cue.start >= float(min_cue_seconds) - 0.02 for cue in cues
        )
        emphasis = [
            item["emphasis"]
            for item in transcript.get("cues", [])
            if item.get("emphasis")
        ]
        checks["emphasis_is_inside_its_cue"] = all(
            transcript["cues"][item["cue_index"]]["start"] <= item["start"]
            <= item["end"] <= transcript["cues"][item["cue_index"]]["end"] + 1e-6
            for item in emphasis
        )
    draft_path_text = storyboard.get("draft_path")
    draft_summary: dict[str, Any] | None = None
    if draft_path_text:
        draft_path = Path(draft_path_text)
        draft_content_path = draft_path / "draft_content.json"
        draft_meta_path = draft_path / "draft_meta_info.json"
        checks["jianying_draft_content_exists"] = draft_content_path.is_file()
        checks["jianying_draft_meta_exists"] = draft_meta_path.is_file()
        try:
            draft_content = json.loads(draft_content_path.read_text(encoding="utf-8"))
            tracks_by_type = {
                track_type: [
                    track
                    for track in draft_content.get("tracks", [])
                    if track.get("type") == track_type
                ]
                for track_type in ("video", "audio", "text")
            }
            video_segments = [
                segment
                for track in tracks_by_type["video"]
                for segment in track.get("segments", [])
            ]
            audio_segments = [
                segment
                for track in tracks_by_type["audio"]
                for segment in track.get("segments", [])
            ]
            text_segments = [
                segment
                for track in tracks_by_type["text"]
                for segment in track.get("segments", [])
            ]
            subtitle_tracks = [
                track
                for track in tracks_by_type["text"]
                if track.get("name") == "subtitles"
            ]
            disclosure_tracks = [
                track
                for track in tracks_by_type["text"]
                if track.get("name") == "ai_disclosure"
            ]
            subtitle_segments = [
                segment
                for track in subtitle_tracks
                for segment in track.get("segments", [])
            ]
            disclosure_segments = [
                segment
                for track in disclosure_tracks
                for segment in track.get("segments", [])
            ]
            expected_text_track_count = 2 if ai_disclosure_required else 1
            checks["jianying_has_video_audio_text_tracks"] = (
                len(tracks_by_type["video"]) == 1
                and len(tracks_by_type["audio"]) == 1
                and len(tracks_by_type["text"]) == expected_text_track_count
            )
            checks["jianying_video_shot_count_matches"] = (
                len(video_segments) == len(storyboard["shots"])
            )
            checks["jianying_has_one_synced_narration"] = (
                len(audio_segments) == 1
                and abs(
                    int(audio_segments[0]["target_timerange"]["duration"])
                    / 1_000_000
                    - audio_duration
                )
                <= 0.2
            )
            checks["jianying_text_timeline_is_editable"] = (
                len(subtitle_tracks) == 1 and len(subtitle_segments) >= len(cues)
            )
            checks["jianying_ai_disclosure_is_separate_and_synced"] = (
                not ai_disclosure_required
                or (
                    len(disclosure_tracks) == 1
                    and len(disclosure_segments) == 1
                    and abs(
                        (
                            int(disclosure_segments[0]["target_timerange"]["start"])
                            + int(disclosure_segments[0]["target_timerange"]["duration"])
                        )
                        / 1_000_000
                        - audio_duration
                    )
                    <= 0.02
                )
            )

            text_materials = draft_content.get("materials", {}).get("texts", [])
            parsed_text_materials: list[dict[str, Any]] = []
            fill_colors: set[tuple[float, ...]] = set()
            for material in text_materials:
                content = json.loads(material.get("content", "{}"))
                parsed_text_materials.append(content)
                for style in content.get("styles", []):
                    color = (
                        style.get("fill", {})
                        .get("content", {})
                        .get("solid", {})
                        .get("color")
                    )
                    if color:
                        fill_colors.add(tuple(round(float(value), 4) for value in color))
            checks["jianying_text_materials_match_segments"] = (
                len(text_materials) == len(text_segments)
                and all(content.get("text") for content in parsed_text_materials)
            )
            transcript_has_emphasis = bool(
                transcript
                and any(cue.get("emphasis") for cue in transcript.get("cues", []))
            )
            checks["jianying_contains_keyword_highlight_color"] = (
                not transcript_has_emphasis or len(fill_colors) >= 2
            )

            video_materials = draft_content.get("materials", {}).get("videos", [])
            audio_materials = draft_content.get("materials", {}).get("audios", [])
            material_paths = [
                Path(str(material.get("path", "")))
                for material in [*video_materials, *audio_materials]
            ]
            checks["jianying_referenced_media_exists"] = (
                len(video_materials) == len(video_segments)
                and len(audio_materials) == len(audio_segments)
                and all(path.is_file() for path in material_paths)
            )
            draft_summary = {
                "path": str(draft_path),
                "video_tracks": len(tracks_by_type["video"]),
                "audio_tracks": len(tracks_by_type["audio"]),
                "text_tracks": len(tracks_by_type["text"]),
                "subtitle_tracks": len(subtitle_tracks),
                "ai_disclosure_tracks": len(disclosure_tracks),
                "video_segments": len(video_segments),
                "audio_segments": len(audio_segments),
                "text_segments": len(text_segments),
                "subtitle_segments": len(subtitle_segments),
                "ai_disclosure_segments": len(disclosure_segments),
                "text_fill_colors": [list(color) for color in sorted(fill_colors)],
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            checks["jianying_draft_structure_is_readable"] = False

    result = {
        "success": all(checks.values()),
        "checks": checks,
        "preview": {
            "path": str(preview.path),
            "width": preview.width,
            "height": preview.height,
            "fps": preview.fps,
            "duration": preview.duration,
            "video_codec": preview.codec_name,
            "audio_codec": preview.audio_codec,
            "pixel_format": preview.pix_fmt,
            "color_range": preview.color_range,
        },
        "audio_duration": audio_duration,
        "shot_count": len(storyboard["shots"]),
        "subtitle_count": len(cues),
        "draft": draft_summary,
    }
    _write_json(run_dir / "validation.json", result)
    return result


def _build_input_hash(
    config: dict[str, Any],
    script_path: Path,
    raw_dir: Path,
    *,
    skip_draft: bool,
    draft_root: Path,
    visual_mode: str,
) -> str:
    media_signature = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
        }
        for item in sorted(raw_dir.iterdir(), key=lambda path: path.name.casefold())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    ] if visual_mode == "local" and raw_dir.is_dir() else []
    payload = {
        "config": config,
        "script": script_path.read_text(encoding="utf-8"),
        "profile_snapshot": (
            json.loads((script_path.parent / "profile_snapshot.json").read_text(encoding="utf-8"))
            if (script_path.parent / "profile_snapshot.json").is_file()
            else None
        ),
        "reuse_source_snapshot": (
            json.loads((script_path.parent / "reuse_source_snapshot.json").read_text(encoding="utf-8"))
            if (script_path.parent / "reuse_source_snapshot.json").is_file()
            else None
        ),
        "media": media_signature,
        "skip_draft": skip_draft,
        "draft_root": str(draft_root.resolve()),
        "visual_mode": visual_mode,
        "pipeline": "stage-7-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _legacy_transcript(
    text: str,
    cues: Sequence[Cue],
    audio_duration: float,
    subtitle_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "normalized_script": text,
        "audio_duration": audio_duration,
        "alignment": {
            "provider": "edge-boundary-legacy",
            "model": None,
            "device": None,
            "coverage": 1.0,
            "minimum_coverage": 1.0,
            "recognized_text": "".join(cue.text for cue in cues),
        },
        "limits": {
            "max_chars_per_line": int(subtitle_config.get("max_chars_per_line", 14)),
            "max_lines": int(subtitle_config.get("max_lines", 2)),
            "min_cue_seconds": 0.0,
            "max_cue_seconds": max((cue.end - cue.start for cue in cues), default=0.0),
        },
        "words": [],
        "cues": [
            {
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "script_start": 0,
                "script_end": 0,
                "emphasis": None,
            }
            for cue in cues
        ],
    }


def build_project(
    project: str,
    *,
    draft_root_override: str | None = None,
    skip_draft: bool = False,
    open_output: bool = False,
    resume: str | None = None,
    visual_mode: str | None = None,
) -> Path:
    configure_pipeline_paths()
    project_dir = _resolve_project(project)
    config = _load_config(project_dir)
    configured_visual_mode = str(config.get("visuals", {}).get("mode", "local"))
    active_visual_mode = str(visual_mode or configured_visual_mode)
    if active_visual_mode not in {"local", "sourced"}:
        raise BuildError("visual mode 只能是 sourced 或 local。")
    subtitle_config = resolve_subtitle_config(config["subtitles"])
    script_path, raw_dir, draft_root = _preflight(
        project_dir, config, draft_root_override, skip_draft, active_visual_mode
    )
    text = script_path.read_text(encoding="utf-8").strip()
    if not text:
        raise BuildError("文案文件为空。")
    input_hash = _build_input_hash(
        config,
        script_path,
        raw_dir,
        skip_draft=skip_draft,
        draft_root=draft_root,
        visual_mode=active_visual_mode,
    )
    output_root = get_studio_paths(ROOT).output_root
    output_root.mkdir(parents=True, exist_ok=True)

    auto_resume = resume == "auto"
    selected_run_dir: Path | None = None
    if resume:
        try:
            selected_run_dir = find_task_dir(
                output_root,
                "latest" if auto_resume else resume,
                project_id=str(config["project"]["id"]),
                draft_root=draft_root,
                input_hash=input_hash if auto_resume else None,
            )
        except FileNotFoundError as exc:
            if not auto_resume:
                raise BuildError(str(exc)) from exc
    if selected_run_dir is not None:
        run_dir = selected_run_dir
        task = TaskState.load(run_dir / "task.json")
        if task.data.get("input_hash") != input_hash:
            raise BuildError(
                "项目配置、文案或素材自该任务失败后已经变化，不能续跑旧任务；"
                "再次双击 run.bat 会自动创建兼容的新任务。"
            )
        run_id = str(task.data["run_id"])
        print(f"[续跑] {task.data['task_id']}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_dir = output_root / f"{config['project']['id']}-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "working").mkdir()
        reuse_config = config.get("visuals", {}).get("reuse", {})
        reuse_enabled = bool(reuse_config.get("enabled")) and active_visual_mode == "sourced"
        task = TaskState.create(
            run_dir / "task.json",
            task_id=run_dir.name,
            run_id=run_id,
            project_id=str(config["project"]["id"]),
            input_hash=input_hash,
            options={
                "draft_root": str(draft_root),
                "skip_draft": skip_draft,
                "visual_mode": active_visual_mode,
            },
            stage_order=REUSE_TASK_STAGES if reuse_enabled else TASK_STAGES,
        )

    reuse_config = config.get("visuals", {}).get("reuse", {})
    reuse_enabled = bool(reuse_config.get("enabled")) and active_visual_mode == "sourced"
    if reuse_enabled and tuple(task.stage_order) != REUSE_TASK_STAGES:
        raise BuildError("派生项目必须使用 18 阶段任务；当前续跑任务与项目复用配置不兼容。")
    stage_positions = {name: index for index, name in enumerate(task.stage_order, 1)}
    stage_total = len(task.stage_order)

    def stage_label(name: str) -> str:
        return f"[{stage_positions[name]}/{stage_total}]"

    phase = "preflight"
    draft_path: Path | None = None
    tts_cache_hit = False
    alignment_cache_hit = False
    media: list[MediaInfo] = []
    excluded: list[str] = []
    asset_audit: dict[str, Any] = {
        "asset_rights_ready": False,
        "human_reviewed": False,
        "ai_disclosure_required": False,
        "provider_counts": {},
    }
    manifest: dict[str, Any] | None = None
    disclosure: dict[str, Any] | None = None
    voice_resolved: dict[str, Any] = resolve_voice_config(config["voice"])
    try:
        if task.can_reuse("preflight", input_hash):
            print(f"{stage_label('preflight')} 复用：前置检查")
        else:
            print(f"{stage_label('preflight')} 检查 FFmpeg、文案、素材与任务输入...")
            task.begin("preflight", input_hash)
            preflight_artifacts = [script_path]
            if active_visual_mode == "local":
                preflight_artifacts.append(raw_dir)
            task.succeed("preflight", input_hash, preflight_artifacts)

        phase = "voice"
        narration_raw = run_dir / "narration.raw.mp3"
        narration = run_dir / "narration.mp3"
        metadata = run_dir / "captions_metadata.jsonl"
        if task.can_reuse("voice", input_hash):
            print(f"{stage_label('voice')} 复用：中文配音")
            tts_cache_hit = True
        else:
            print(f"{stage_label('voice')} 生成或复用 MoneyPrinterTurbo 中文配音...")
            task.begin("voice", input_hash)
            narration_raw, narration, metadata, tts_cache_hit, voice_resolved = create_or_reuse_tts(
                text, config["voice"], run_dir
            )
            task.succeed("voice", input_hash, [narration_raw, narration, metadata])
        audio_duration = probe_audio_duration(narration)

        phase = "alignment"
        alignment_provider = str(
            subtitle_config.get("alignment_provider", "edge_metadata")
        )
        alignment_path = run_dir / "working" / "alignment.json"
        alignment: dict[str, Any] | None = None
        if alignment_provider == "moneyprinter_whisper":
            if task.can_reuse("alignment", input_hash):
                print(f"{stage_label('alignment')} 复用：Whisper 逐词时间轴")
                alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
                alignment_cache_hit = True
            else:
                print(f"{stage_label('alignment')} 使用本地 Whisper large-v3 对齐逐词时间轴...")
                task.begin("alignment", input_hash)
                initial_prompt = whisper_initial_prompt(
                    text,
                    voice_resolved,
                    subtitle_config,
                )
                alignment, alignment_path, alignment_cache_hit = create_or_reuse_alignment(
                    narration,
                    subtitle_config,
                    run_dir,
                    initial_prompt=initial_prompt,
                )
                task.succeed("alignment", input_hash, [alignment_path])
        elif alignment_provider == "edge_metadata":
            print(f"{stage_label('alignment')} 兼容模式：使用 Edge 边界时间轴")
            alignment = edge_metadata_alignment(metadata)
            if not task.can_reuse("alignment", input_hash):
                task.begin("alignment", input_hash)
                task.succeed("alignment", input_hash, [metadata])
        else:
            raise BuildError(f"未知字幕对齐提供商：{alignment_provider}")

        phase = "captions"
        captions = run_dir / "captions.srt"
        captions_ass = run_dir / "captions.ass"
        transcript_path = run_dir / "transcript.json"
        alignment_diagnostics_path = run_dir / "alignment_diagnostics.json"
        emphasis_path = run_dir / "emphasis.json"
        font_dir = run_dir / "working" / "fonts"
        if task.can_reuse("captions", input_hash):
            print(f"{stage_label('captions')} 复用：稳定底稿和局部强调字幕")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            cues = cues_from_transcript(transcript)
        else:
            print(f"{stage_label('captions')} 生成稳定底稿和局部强调字幕...")
            task.begin("captions", input_hash)
            if alignment is not None:
                try:
                    transcript, cues, emphasis = build_transcript(
                        text,
                        alignment,
                        audio_duration=audio_duration,
                        subtitle_config=subtitle_config,
                    )
                except AlignmentQualityError as exc:
                    _write_json(alignment_diagnostics_path, exc.diagnostics)
                    raise BuildError(str(exc)) from exc
                except ValueError as exc:
                    raise BuildError(str(exc)) from exc
            else:
                raise BuildError("字幕阶段缺少可用的对齐数据。")
            _write_json(transcript_path, transcript)
            _write_json(alignment_diagnostics_path, transcript["alignment"])
            _write_json(
                emphasis_path,
                {
                    "schema_version": 1,
                    "mode": subtitle_config.get("emphasis", {}).get("mode", "none"),
                    "events": emphasis,
                },
            )
            write_srt(captions, cues)
            configured_font_file = subtitle_config.get("font_file")
            if configured_font_file:
                font = Path(os.path.expandvars(str(configured_font_file))).resolve()
                if not font.is_file():
                    raise BuildError(f"字幕字体文件不存在：{font}")
            else:
                try:
                    font = ensure_source_han_font(
                        auto_download=bool(subtitle_config.get("font_auto_download", False))
                    )
                except RuntimeSetupError as exc:
                    raise BuildError(f"思源黑体准备失败：{exc}") from exc
            render_subtitle_config = dict(subtitle_config)
            caption_artifacts: list[Path] = [
                captions,
                captions_ass,
                transcript_path,
                alignment_diagnostics_path,
                emphasis_path,
            ]
            if font:
                font_dir.mkdir(parents=True, exist_ok=True)
                installed_font = font_dir / font.name
                shutil.copy2(font, installed_font)
                caption_artifacts.append(installed_font)
            else:
                render_subtitle_config["font_name"] = subtitle_config.get(
                    "font_fallback", "Microsoft YaHei"
                )
                print(
                    "      [提示] 未检测到本地思源黑体，本次预览使用系统字体；"
                    "手动安装说明见 README.md 的“模型和字体的手动准备”。"
                )
            write_ass(
                captions_ass,
                cues,
                config["canvas"],
                render_subtitle_config,
                transcript,
            )
            task.succeed(
                "captions",
                input_hash,
                caption_artifacts,
            )

        scheduled_shots = schedule_shots(
            duration=audio_duration,
            fps=int(config["canvas"]["fps"]),
            hook_until=float(config["pacing"]["hook_until_seconds"]),
            hook_range=tuple(float(value) for value in config["pacing"]["hook_shot_seconds"]),
            body_range=tuple(float(value) for value in config["pacing"]["body_shot_seconds"]),
            seed=int(config.get("media", {}).get("seed", 20260810)),
        )
        for shot in scheduled_shots:
            shot["caption_text"] = _caption_for_shot(shot, cues)

        if active_visual_mode == "sourced":
            visuals_config = config.get("visuals", {})
            visual_strategy = str(
                visuals_config.get("strategy", "museum_and_ai")
            ).lower()
            search_config = visuals_config.get("search", {})
            env = load_local_env(ROOT)
            provenance_dir = run_dir / "provenance"
            provenance_dir.mkdir(parents=True, exist_ok=True)

            phase = "visual_plan"
            scene_plan_path = run_dir / "scene_plan.json"
            planner_provider = str(
                visuals_config.get("planner", {}).get("provider", "openai")
            ).lower()
            response_snapshot = provenance_dir / f"{planner_provider}-scene-plan-response.json"
            plan_audit_path = run_dir / "scene_plan_audit.json"
            plan_cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "script_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "shots": scheduled_shots,
                        "planner": visuals_config.get("planner", {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            plan_cache = get_studio_paths(ROOT).cache_root / "visuals" / "scene-plan" / f"{plan_cache_key}.json"
            plan_audit_cache = (
                get_studio_paths(ROOT).cache_root / "visuals" / "scene-plan" / f"{plan_cache_key}-audit.json"
            )
            if task.can_reuse("visual_plan", input_hash):
                print(f"{stage_label('visual_plan')} 复用：动态时代约束和镜头视觉意图")
                scene_plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
            else:
                print(f"{stage_label('visual_plan')} 用 {planner_provider} 生成动态时代约束并做语义审校...")
                task.begin("visual_plan", input_hash)
                if plan_cache.is_file() and plan_audit_cache.is_file():
                    scene_plan = json.loads(plan_cache.read_text(encoding="utf-8"))
                    plan_audit = json.loads(plan_audit_cache.read_text(encoding="utf-8"))
                    _write_json(response_snapshot, {"cache_hit": True, "cache_key": plan_cache_key})
                else:
                    try:
                        scene_plan, safe_response, plan_audit = create_visual_plan_with_audit(
                            text,
                            scheduled_shots,
                            visuals_config.get("planner", {}),
                            env,
                            diagnostics_path=response_snapshot,
                        )
                    except VisualPlannerError as exc:
                        raise BuildError(str(exc)) from exc
                    _write_json(response_snapshot, safe_response)
                    if plan_audit.get("status") in {"reviewed", "disabled"}:
                        plan_cache.parent.mkdir(parents=True, exist_ok=True)
                        _write_json(plan_cache, scene_plan)
                        _write_json(plan_audit_cache, plan_audit)
                _write_json(scene_plan_path, scene_plan)
                _write_json(plan_audit_path, plan_audit)
                task.succeed(
                    "visual_plan",
                    input_hash,
                    [scene_plan_path, plan_audit_path, response_snapshot],
                )

            reuse_snapshot: dict[str, Any] | None = None
            reuse_selection: dict[str, Any] | None = None
            supply_scene_plan = scene_plan
            if reuse_enabled:
                snapshot_name = str(
                    config.get("revision", {}).get(
                        "reuse_snapshot_file", "reuse_source_snapshot.json"
                    )
                )
                if Path(snapshot_name).name != snapshot_name:
                    raise BuildError("复用快照文件名无效。")
                snapshot_path = project_dir / snapshot_name
                if not snapshot_path.is_file():
                    raise BuildError("派生项目缺少不可变 reuse_source_snapshot.json。")
                reuse_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

                phase = "asset_reuse_match"
                reuse_plan_path = run_dir / "asset_reuse_plan.json"
                reuse_cache_key = hashlib.sha256(
                    json.dumps(
                        {
                            "scene_plan": scene_plan,
                            "snapshot_sha256": reuse_snapshot.get("sha256"),
                            "reuse": reuse_config,
                            "planner": visuals_config.get("planner", {}),
                            "prompt_version": "asset-reuse-match-v1",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                reuse_cache = (
                    get_studio_paths(ROOT).cache_root
                    / "visuals"
                    / "reuse-match"
                    / f"{reuse_cache_key}.json"
                )
                if task.can_reuse("asset_reuse_match", input_hash):
                    print(f"{stage_label('asset_reuse_match')} 复用：旧画面匹配结果")
                    reuse_plan = json.loads(reuse_plan_path.read_text(encoding="utf-8"))
                else:
                    print(f"{stage_label('asset_reuse_match')} 匹配父成片已选素材与未选 AI 候选...")
                    task.begin("asset_reuse_match", input_hash)
                    try:
                        if reuse_cache.is_file():
                            reuse_plan = json.loads(reuse_cache.read_text(encoding="utf-8"))
                        else:
                            reuse_plan = build_reuse_plan(
                                scene_plan,
                                reuse_snapshot,
                                reuse_config,
                                visuals_config.get("planner", {}),
                                env,
                                get_studio_paths(ROOT).cache_root,
                            )
                            reuse_cache.parent.mkdir(parents=True, exist_ok=True)
                            _write_json(reuse_cache, reuse_plan)
                    except AssetReuseError as exc:
                        raise BuildError(str(exc)) from exc
                    _write_json(reuse_plan_path, reuse_plan)
                    task.succeed("asset_reuse_match", input_hash, [reuse_plan_path])

                phase = "asset_reuse_review"
                reuse_selection_path = run_dir / "asset_reuse_selection.json"
                reuse_report_path = run_dir / "reuse_report.json"
                if task.can_reuse("asset_reuse_review", input_hash):
                    print(f"{stage_label('asset_reuse_review')} 复用：旧画面人工匹配选择")
                    reuse_selection = json.loads(reuse_selection_path.read_text(encoding="utf-8"))
                else:
                    print(f"{stage_label('asset_reuse_review')} 等待旧画面匹配审核...")
                    task.begin("asset_reuse_review", input_hash)
                    while not reuse_selection_path.is_file():
                        task.wait_for_review(
                            "asset_reuse_review", input_hash, [reuse_plan_path, snapshot_path]
                        )
                        review_status = run_reuse_review_server(
                            run_dir,
                            reuse_plan,
                            reuse_snapshot,
                            reuse_config,
                            get_studio_paths(ROOT).cache_root,
                            open_browser=True,
                        )
                        if review_status != "submitted":
                            raise BuildError("画面复用审核尚未提交；任务保持 waiting_for_review。")
                    reuse_selection = json.loads(reuse_selection_path.read_text(encoding="utf-8"))
                    task.succeed(
                        "asset_reuse_review",
                        input_hash,
                        [reuse_selection_path, reuse_report_path],
                    )
                supply_scene_plan = unmatched_scene_plan(scene_plan, reuse_selection)
                # A human reuse choice is a downstream build input. Keep the
                # task-level base hash stable for resume discovery, but salt
                # every later stage so manually changing the selection cannot
                # reuse search, downloads, clips or drafts from the wrong choice.
                input_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "base_input_hash": input_hash,
                            "reuse_snapshot_sha256": reuse_snapshot.get("sha256"),
                            "reuse_selection": reuse_selection,
                            "reuse_policy": reuse_config,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()

            phase = "asset_search"
            candidates_path = run_dir / "asset_candidates.json"
            search_cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "scene_plan": supply_scene_plan,
                        "visual_strategy": visual_strategy,
                        "search": search_config,
                        "credential_capabilities": {
                            "smithsonian": bool(env.get("SMITHSONIAN_API_KEY")),
                            "openverse_token": bool(env.get("OPENVERSE_API_TOKEN")),
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            search_cache = get_studio_paths(ROOT).cache_root / "visuals" / "search" / f"{search_cache_key}.json"
            if task.can_reuse("asset_search", input_hash):
                label = "纯 AI 空候选清单" if visual_strategy == "ai_only" else "馆藏搜索结果"
                print(f"{stage_label('asset_search')} 复用：{label}")
                candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
            else:
                task.begin("asset_search", input_hash)
                if visual_strategy == "ai_only":
                    print(
                        f"{stage_label('asset_search')} 已选择纯 AI：跳过 The Met、Smithsonian、"
                        "Wikimedia 和 Openverse 网络搜索..."
                    )
                    candidates = build_ai_only_candidates(
                        supply_scene_plan, search_config
                    )
                elif reuse_enabled and not supply_scene_plan.get("shots"):
                    print(f"{stage_label('asset_search')} 全部镜头已复用：跳过馆藏网络搜索...")
                    candidates = build_ai_only_candidates(supply_scene_plan, search_config)
                    candidates["visual_strategy"] = visual_strategy
                    candidates["search_skip_reason"] = "all_shots_reused"
                    for status in candidates.get("provider_status", {}).values():
                        status["message"] = "全部新镜头已确认复用旧画面，未发起馆藏网络搜索。"
                else:
                    print(f"{stage_label('asset_search')} 搜索 The Met、Smithsonian、Wikimedia 和 Openverse...")
                    if search_cache.is_file():
                        candidates = json.loads(search_cache.read_text(encoding="utf-8"))
                    else:
                        try:
                            candidates = build_search_results(supply_scene_plan, search_config, env)
                        except VisualSupplyError as exc:
                            raise BuildError(str(exc)) from exc
                        if not any(
                            item.get("status") == "failed"
                            for item in candidates.get("provider_status", {}).values()
                        ):
                            search_cache.parent.mkdir(parents=True, exist_ok=True)
                            _write_json(search_cache, candidates)
                    candidates = prepare_and_deduplicate(
                        run_dir,
                        candidates,
                        int(search_config.get("recommendation_threshold", 70)),
                    )
                _write_json(candidates_path, candidates)
                task.succeed("asset_search", input_hash, [candidates_path])

            phase = "asset_semantic_review"
            semantic_review_path = run_dir / "asset_semantic_review.json"
            if task.can_reuse("asset_semantic_review", input_hash):
                label = "纯 AI 无馆藏候选" if visual_strategy == "ai_only" else "DeepSeek 馆藏候选语义复核"
                print(f"{stage_label('asset_semantic_review')} 复用：{label}")
                candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
                semantic_review = json.loads(
                    semantic_review_path.read_text(encoding="utf-8")
                )
            else:
                task.begin("asset_semantic_review", input_hash)
                if reuse_enabled and not supply_scene_plan.get("shots"):
                    print(f"{stage_label('asset_semantic_review')} 全部镜头已复用：跳过馆藏候选语义复核...")
                    semantic_review = {
                        "schema_version": 1,
                        "status": "not_applicable",
                        "reason": "all_shots_reused",
                        "provider": None,
                        "batches": [],
                    }
                    semantic_artifacts = []
                elif visual_strategy == "ai_only":
                    print(f"{stage_label('asset_semantic_review')} 已选择纯 AI：没有馆藏候选，跳过候选语义复核...")
                    semantic_review = {
                        "schema_version": 1,
                        "status": "not_applicable",
                        "reason": "ai_only",
                        "provider": None,
                        "batches": [],
                    }
                    semantic_artifacts = []
                else:
                    print(f"{stage_label('asset_semantic_review')} 用 DeepSeek 批量复核候选的时代、主题和画面相关性...")
                    try:
                        candidates, semantic_review, semantic_artifacts = review_asset_candidates(
                            supply_scene_plan,
                            candidates,
                            visuals_config.get("planner", {}),
                            search_config,
                            env,
                            get_studio_paths(ROOT).cache_root,
                        )
                    except SemanticReviewError as exc:
                        raise BuildError(str(exc)) from exc
                _write_json(candidates_path, candidates)
                _write_json(semantic_review_path, semantic_review)
                task.succeed(
                    "asset_semantic_review",
                    input_hash,
                    [candidates_path, semantic_review_path, *semantic_artifacts],
                )

            phase = "ai_fallback"
            if task.can_reuse("ai_fallback", input_hash):
                print(f"{stage_label('ai_fallback')} 复用：AI 候选")
                candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
            else:
                ai_provider = str(
                    visuals_config.get("ai_fallback", {}).get("provider", "openai")
                ).lower()
                ai_count = int(
                    visuals_config.get("ai_fallback", {}).get(
                        "candidates_per_shot", 1
                    )
                )
                print(
                    f"{stage_label('ai_fallback')} 为每个待补镜头准备 {ai_count} 张 {ai_provider} "
                    "AI 候选并检查额度..."
                )
                task.begin("ai_fallback", input_hash)
                try:
                    if reuse_enabled and not supply_scene_plan.get("shots"):
                        generated = []
                    else:
                        candidates, generated = add_ai_fallbacks(
                            candidates,
                            supply_scene_plan,
                            visuals_config.get("ai_fallback", {}),
                            env,
                            get_studio_paths(ROOT).cache_root,
                            provenance_dir,
                            project_dir,
                            candidates_path,
                        )
                    if reuse_enabled:
                        assert reuse_snapshot is not None and reuse_selection is not None
                        candidates = merge_reused_candidates(
                            scene_plan,
                            candidates,
                            reuse_selection,
                            reuse_snapshot,
                            get_studio_paths(ROOT).cache_root,
                        )
                except VisualSupplyError as exc:
                    raise BuildError(str(exc)) from exc
                except AssetReuseError as exc:
                    raise BuildError(str(exc)) from exc
                _write_json(candidates_path, candidates)
                task.succeed("ai_fallback", input_hash, [candidates_path, *generated])

            phase = "asset_review"
            selection_path = run_dir / "asset_selection.json"
            if task.can_reuse("asset_review", input_hash):
                print(f"{stage_label('asset_review')} 复用：人工素材选择")
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
            else:
                print(f"{stage_label('asset_review')} 等待一次最终素材审核...")
                task.begin("asset_review", input_hash)
                while not selection_path.is_file():
                    task.wait_for_review("asset_review", input_hash, [candidates_path])
                    _write_json(
                        run_dir / "build_report.json",
                        {
                            "status": "waiting_for_review",
                            "run_id": run_id,
                            "task_id": task.data["task_id"],
                            "review_command": f"{sys.executable} -m app assets review --resume {task.data['task_id']}",
                        },
                    )
                    review_status = run_review_server(run_dir, candidates, open_browser=True)
                    if review_status == "retry":
                        request_path = run_dir / "asset_review_request.json"
                        request = json.loads(request_path.read_text(encoding="utf-8"))
                        try:
                            candidates = apply_review_request(
                                candidates,
                                scene_plan,
                                request,
                                visuals_config,
                                env,
                                get_studio_paths(ROOT).cache_root,
                                provenance_dir,
                                project_dir,
                            )
                        except VisualSupplyError as exc:
                            raise BuildError(str(exc)) from exc
                        candidates = prepare_and_deduplicate(
                            run_dir,
                            candidates,
                            int(search_config.get("recommendation_threshold", 70)),
                        )
                        try:
                            candidates, semantic_review, _ = review_asset_candidates(
                                scene_plan,
                                candidates,
                                visuals_config.get("planner", {}),
                                search_config,
                                env,
                                get_studio_paths(ROOT).cache_root,
                            )
                        except SemanticReviewError as exc:
                            raise BuildError(str(exc)) from exc
                        _write_json(candidates_path, candidates)
                        _write_json(semantic_review_path, semantic_review)
                        print("      候选已更新，重新打开最终审核页...")
                        continue
                    if review_status != "submitted":
                        raise BuildError("素材审核尚未提交；任务保持 waiting_for_review。")
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                task.succeed("asset_review", input_hash, [selection_path])

            input_hash = hashlib.sha256(
                json.dumps(
                    {
                        "upstream_input_hash": input_hash,
                        "asset_selection": selection,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

            phase = "asset_download"
            manifest_path = run_dir / "assets_manifest.json"
            reuse_asset_download = task.can_reuse("asset_download", input_hash)
            if reuse_asset_download:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                try:
                    validate_asset_manifest(manifest, search_config)
                except VisualSupplyError as exc:
                    reuse_asset_download = False
                    task.invalidate_after("asset_download")
                    print(f"{stage_label('asset_download')} 已发现旧素材损坏，将重新下载：{exc}")
                else:
                    print(f"{stage_label('asset_download')} 复用：内容寻址素材缓存（完整解码已通过）")
            if not reuse_asset_download:
                print(f"{stage_label('asset_download')} 复核授权并下载审核通过的原图...")
                task.begin("asset_download", input_hash)
                try:
                    manifest = download_selected_assets(
                        selection, search_config, env, get_studio_paths(ROOT).cache_root, run_dir
                    )
                except VisualSupplyError as exc:
                    raise BuildError(str(exc)) from exc
                asset_files = [Path(item["local_path"]) for item in manifest["assets"]]
                task.succeed("asset_download", input_hash, [manifest_path, *asset_files])

            phase = "license_audit"
            audit_path = run_dir / "license_audit.json"
            if task.can_reuse("license_audit", input_hash):
                print(f"{stage_label('license_audit')} 复用：许可证台账")
                asset_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            else:
                print(f"{stage_label('license_audit')} 生成许可证台账、Credits 和 AI 说明...")
                task.begin("license_audit", input_hash)
                try:
                    licenses, credits, ai_disclosure, asset_audit = write_license_outputs(
                        manifest, run_dir
                    )
                except VisualSupplyError as exc:
                    raise BuildError(str(exc)) from exc
                task.succeed(
                    "license_audit",
                    input_hash,
                    [audit_path, licenses, credits, ai_disclosure, provenance_dir],
                )
            disclosure = {
                "required": bool(asset_audit.get("ai_disclosure_required")),
                "text": visuals_config.get("review", {}).get(
                    "disclosure_text", "部分画面为 AI 历史重构"
                ),
                "seconds": float(
                    visuals_config.get("review", {}).get("disclosure_seconds", 2.0)
                ),
                "end": audio_duration,
            }
            render_subtitle_config = dict(subtitle_config)
            if not font_dir.is_dir() or not any(font_dir.iterdir()):
                render_subtitle_config["font_name"] = subtitle_config.get(
                    "font_fallback", "Microsoft YaHei"
                )
            write_ass(
                captions_ass,
                cues,
                config["canvas"],
                render_subtitle_config,
                transcript,
                disclosure=disclosure,
            )
        else:
            for stage in (
                "visual_plan", "asset_search", "asset_semantic_review", "ai_fallback", "asset_review",
                "asset_download", "license_audit",
            ):
                if not task.can_reuse(stage, input_hash):
                    task.begin(stage, input_hash)
                    task.succeed(stage, input_hash)

        phase = "storyboard"
        storyboard_path = run_dir / "storyboard.json"
        if task.can_reuse("storyboard", input_hash):
            print(f"{stage_label('storyboard')} 复用：storyboard v2")
            storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
            if active_visual_mode == "local":
                media, excluded = _list_media(raw_dir, config["media"])
        else:
            print(f"{stage_label('storyboard')} 生成 storyboard v2...")
            task.begin("storyboard", input_hash)
            if active_visual_mode == "sourced":
                assert manifest is not None
                try:
                    storyboard = build_sourced_storyboard(
                        config["project"]["title"],
                        audio_duration,
                        config["canvas"],
                        scheduled_shots,
                        manifest,
                    )
                except VisualSupplyError as exc:
                    raise BuildError(str(exc)) from exc
            else:
                media, excluded = _list_media(raw_dir, config["media"])
                storyboard = build_storyboard(
                    config["project"]["title"],
                    audio_duration,
                    config["canvas"],
                    config["pacing"],
                    config["media"],
                    media,
                    cues,
                )
            _write_json(storyboard_path, storyboard)
            task.succeed("storyboard", input_hash, [storyboard_path])

        phase = "clips"
        clips = [run_dir / shot["rendered_clip"] for shot in storyboard["shots"]]
        if task.can_reuse("clips", input_hash):
            print(f"{stage_label('clips')} 复用：预渲染竖屏镜头")
        else:
            print(f"{stage_label('clips')} 生成图片代理并预渲染竖屏镜头...")
            task.begin("clips", input_hash)
            proxies = _prepare_image_proxies(storyboard, run_dir)
            clips = []
            for index, shot in enumerate(storyboard["shots"], 1):
                print(
                    f"      镜头 {index:02d}/{len(storyboard['shots']):02d}: "
                    f"{shot['source_name']} ({shot['fit']})"
                )
                clips.append(
                    render_shot(
                        shot,
                        run_dir,
                        config["canvas"],
                        proxies,
                        get_studio_paths(ROOT).cache_root / "render",
                    )
                )
            task.succeed("clips", input_hash, clips)

        phase = "preview"
        preview = run_dir / config["output"]["preview_filename"]
        if task.can_reuse("preview", input_hash):
            print(f"{stage_label('preview')} 复用：字幕预览 MP4")
        else:
            print(f"{stage_label('preview')} 合并镜头并渲染字幕预览 MP4...")
            task.begin("preview", input_hash)
            visuals = _concat_clips(clips, run_dir)
            preview = _render_preview(
                visuals,
                narration,
                captions_ass,
                run_dir,
                config["output"]["preview_filename"],
                fonts_dir=font_dir if font_dir.is_dir() else None,
            )
            task.succeed("preview", input_hash, [preview, visuals])

        phase = "draft"
        if skip_draft:
            print(f"{stage_label('draft')} 已按参数跳过剪映草稿。")
            if not task.can_reuse("draft", input_hash):
                task.begin("draft", input_hash)
                task.succeed("draft", input_hash)
        elif task.can_reuse("draft", input_hash):
            print(f"{stage_label('draft')} 复用：剪映可编辑富文本草稿")
            draft_path_text = storyboard.get("draft_path")
            draft_path = Path(draft_path_text) if draft_path_text else None
        else:
            print(f"{stage_label('draft')} 生成剪映可编辑富文本草稿...")
            task.begin("draft", input_hash)
            configured_root = Path(
                str(config.get("jianying", {}).get("draft_root") or discover_jianying_draft_root())
            ).expanduser().resolve()
            if _same_path(draft_root, configured_root) and _jianying_running():
                raise BuildError(
                    "检测到剪映专业版正在运行。前面的产物已经保留；关闭剪映后再次双击 "
                    "run.bat，即可自动从草稿阶段继续。"
                )
            draft_name = f"{config['jianying']['draft_name_prefix']}-{run_id}"
            draft_path = _create_jianying_draft(
                draft_root,
                draft_name,
                storyboard,
                run_dir,
                narration,
                transcript,
                subtitle_config,
                disclosure=disclosure,
            )
            storyboard["draft_path"] = str(draft_path)
            _write_json(storyboard_path, storyboard)
            task.succeed("draft", input_hash, [draft_path])

        phase = "validation"
        if task.can_reuse("validation", input_hash):
            print(f"{stage_label('validation')} 复用：自动验收")
            validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
        else:
            print(f"{stage_label('validation')} 验证分辨率、时长、字幕、授权追踪和草稿结构...")
            task.begin("validation", input_hash)
            validation = validate_run(run_dir)
            if not validation["success"]:
                failed = [name for name, passed in validation["checks"].items() if not passed]
                raise BuildError("验收检查未通过：" + "、".join(failed))
            task.succeed("validation", input_hash, [run_dir / "validation.json"])

        report = {
            "status": "success",
            "run_id": run_id,
            "task_id": task.data["task_id"],
            "project": config["project"],
            "created_at": datetime.now().astimezone().isoformat(),
            "publish_ready": False,
            "warning": (
                "馆藏授权和 AI 标识已生成，但最终史实核对与平台发布标识仍需人工完成。"
                if active_visual_mode == "sourced"
                else "当前本地素材只用于技术实验，来源和授权未核验，不可直接发布。"
            ),
            "visual_mode": active_visual_mode,
            "visual_strategy": (
                str(config.get("visuals", {}).get("strategy", "museum_and_ai"))
                if active_visual_mode == "sourced"
                else "local"
            ),
            "revision": config.get("revision"),
            **asset_audit,
            "voice": voice_resolved,
            "tts_cache_hit": tts_cache_hit,
            "alignment_cache_hit": alignment_cache_hit,
            "moneyprinterturbo": {"version": MPT_VERSION, "commit": MPT_COMMIT},
            "dependencies": _package_versions(),
            "source_materials": (
                [
                    {
                        "path": str(item.path),
                        "kind": item.kind,
                        "width": item.width,
                        "height": item.height,
                        "duration": item.duration,
                    }
                    for item in media
                ]
                if active_visual_mode == "local"
                else (manifest or {}).get("assets", [])
            ),
            "excluded_materials": excluded,
            "outputs": {
                "preview": str(preview),
                "narration_raw": str(narration_raw),
                "narration": str(narration),
                "captions": str(captions),
                "captions_ass": str(captions_ass),
                "transcript": str(transcript_path),
                "emphasis": str(emphasis_path),
                "storyboard": str(storyboard_path),
                "scene_plan": str(run_dir / "scene_plan.json") if active_visual_mode == "sourced" else None,
                "scene_plan_audit": str(run_dir / "scene_plan_audit.json") if active_visual_mode == "sourced" else None,
                "asset_reuse_plan": str(run_dir / "asset_reuse_plan.json") if reuse_enabled else None,
                "asset_reuse_selection": str(run_dir / "asset_reuse_selection.json") if reuse_enabled else None,
                "reuse_report": str(run_dir / "reuse_report.json") if reuse_enabled else None,
                "asset_candidates": str(run_dir / "asset_candidates.json") if active_visual_mode == "sourced" else None,
                "asset_semantic_review": str(run_dir / "asset_semantic_review.json") if active_visual_mode == "sourced" else None,
                "asset_selection": str(run_dir / "asset_selection.json") if active_visual_mode == "sourced" else None,
                "assets_manifest": str(run_dir / "assets_manifest.json") if active_visual_mode == "sourced" else None,
                "licenses": str(run_dir / "licenses.csv") if active_visual_mode == "sourced" else None,
                "credits": str(run_dir / "CREDITS.md") if active_visual_mode == "sourced" else None,
                "ai_disclosure": str(run_dir / "AI_DISCLOSURE.md") if active_visual_mode == "sourced" else None,
                "task": str(task.path),
                "jianying_draft": str(draft_path) if draft_path else None,
            },
            "validation": validation,
        }
        _write_json(run_dir / "build_report.json", report)
        task.complete()
    except Exception as exc:
        task.fail(phase, str(exc))
        failure = {
            "status": "failed",
            "run_id": run_id,
            "task_id": task.data["task_id"],
            "phase": phase,
            "error": str(exc),
            "created_at": datetime.now().astimezone().isoformat(),
            "resume_command": " ".join(
                [
                    str(sys.executable),
                    "-m app build --project",
                    str(config["project"]["id"]),
                    "--resume",
                    str(task.data["task_id"]),
                    "--visual-mode",
                    active_visual_mode,
                    *(["--draft-root", str(draft_root)] if draft_root_override else []),
                ]
            ),
        }
        _write_json(run_dir / "build_report.json", failure)
        if isinstance(exc, BuildError):
            raise
        raise BuildError(f"{phase} 阶段发生未预期错误：{exc}") from exc

    print(f"输出目录：{run_dir}")
    if draft_path:
        print(f"剪映草稿：{draft_path}")
    if open_output and sys.platform == "win32":
        os.startfile(run_dir)  # type: ignore[attr-defined]
    return run_dir


def review_asset_task(token: str = "latest") -> Path:
    """Open the localhost reviewer for a waiting task, then resume that build."""
    try:
        run_dir = find_task_dir(get_studio_paths(ROOT).output_root, token)
    except FileNotFoundError as exc:
        raise BuildError(str(exc)) from exc
    task = TaskState.load(run_dir / "task.json")
    options = task.data.get("options", {})
    return build_project(
        str(task.data["project_id"]),
        draft_root_override=options.get("draft_root"),
        skip_draft=bool(options.get("skip_draft", False)),
        resume=str(task.data["task_id"]),
        visual_mode=str(options.get("visual_mode", "sourced")),
    )


def create_voice_audition(project: str, *, open_output: bool = False) -> Path:
    project_dir = _resolve_project(project)
    config = _load_config(project_dir)
    script_path = (project_dir / config["project"]["script_file"]).resolve()
    text = re.sub(r"\s+", "", script_path.read_text(encoding="utf-8").strip())
    if not text:
        raise BuildError("文案文件为空。")
    cutoff = min(len(text), 105)
    for punctuation in "。！？!?":
        position = text.rfind(punctuation, 65, cutoff)
        if position >= 0:
            cutoff = position + 1
            break
    audition_text = text[:cutoff]
    profiles = _load_named_config(ROOT / "config" / "voice_profiles.yaml", "profiles")
    profile_ids = [
        "yunxi_current_reference",
        "yunxi_natural",
        "yunxi_calm",
        "yunyang_soft",
        "yunyang_clear",
        "yunjian_story",
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output = get_studio_paths(ROOT).output_root / f"voice-audition-{timestamp}"
    working = output / "working"
    working.mkdir(parents=True, exist_ok=False)
    (output / "audition_text.txt").write_text(audition_text + "\n", encoding="utf-8")
    samples: list[dict[str, Any]] = []
    for index, profile_id in enumerate(profile_ids, 1):
        print(f"[{index}/6] 生成试听：{profiles[profile_id]['label']}")
        sample_working = working / profile_id
        sample_working.mkdir()
        raw, normalized, _, cache_hit, resolved = create_or_reuse_tts(
            audition_text,
            {
                "provider": "moneyprinter_edge",
                "profile": profile_id,
                "pronunciation": config.get("voice", {}).get("pronunciation", {}),
            },
            sample_working,
        )
        filename = f"{index:02d}_{profile_id}.mp3"
        destination = output / filename
        shutil.copy2(normalized, destination)
        samples.append(
            {
                "order": index,
                "profile": profile_id,
                "label": resolved["label"],
                "voice": resolved["voice"],
                "rate": resolved["rate"],
                "pitch": resolved["pitch"],
                "duration": probe_audio_duration(destination),
                "cache_hit": cache_hit,
                "file": filename,
                "raw_file": str(raw),
            }
        )
    _write_json(
        output / "comparison.json",
        {
            "schema_version": 1,
            "project": config["project"]["id"],
            "text": audition_text,
            "moneyprinterturbo": {"version": MPT_VERSION, "commit": MPT_COMMIT},
            "samples": samples,
        },
    )
    (output / "如何选择.txt").write_text(
        "按文件名前的数字依次试听。选中后，把 project.yaml 中 voice.profile "
        "改为对应的 profile 名称即可，无需修改代码。\n",
        encoding="utf-8",
    )
    print(f"试听目录：{output}")
    if open_output and sys.platform == "win32":
        os.startfile(output)  # type: ignore[attr-defined]
    return output
