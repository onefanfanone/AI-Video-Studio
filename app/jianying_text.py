from __future__ import annotations

import copy
import json
from typing import Any


def _rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def _pieces(cue: dict[str, Any]) -> list[tuple[float, float, tuple[int, int] | None]]:
    start = float(cue["start"])
    end = float(cue["end"])
    emphasis = cue.get("emphasis")
    if not emphasis:
        return [(start, end, None)]
    highlight_start = min(end, max(start, float(emphasis["start"])))
    highlight_end = min(end, max(highlight_start, float(emphasis["end"])))
    def actual_index(flat_index: int) -> int:
        visible = 0
        for position, character in enumerate(str(cue["text"])):
            if character == "\n":
                continue
            if visible == flat_index:
                return position
            visible += 1
        return len(str(cue["text"]))

    text_range = (
        actual_index(int(emphasis["text_start"])),
        actual_index(int(emphasis["text_end"])),
    )
    result: list[tuple[float, float, tuple[int, int] | None]] = []
    if highlight_start - start >= 0.01:
        result.append((start, highlight_start, None))
    if highlight_end - highlight_start >= 0.01:
        result.append((highlight_start, highlight_end, text_range))
    if end - highlight_end >= 0.01:
        result.append((highlight_end, end, None))
    return result or [(start, end, None)]


def add_rich_subtitles(
    script: Any,
    draft: Any,
    transcript: dict[str, Any],
    subtitle_config: dict[str, Any],
    disclosure: dict[str, Any] | None = None,
) -> Any:
    track = script.append_track(draft.TrackSpec(draft.TrackType.text, "subtitles"))
    base_color = _rgb(subtitle_config.get("base_color", "#FFFFFF"))
    highlight_color = _rgb(subtitle_config.get("highlight_color", "#FFD54A"))
    outline_color = _rgb(subtitle_config.get("draft_outline_color", "#080808"))
    shadow_color = _rgb(subtitle_config.get("shadow_color", "#000000"))
    draft_size = float(subtitle_config.get("draft_font_size", 9.5))
    transform_y = float(subtitle_config.get("draft_transform_y", -0.55))
    border_width = float(subtitle_config.get("draft_outline", 34.0))
    style_preset = subtitle_config.get("style_preset", "history_keyword")
    draft_font_name = str(
        subtitle_config.get("draft_font_type", "SourceHanSansCN_Bold")
    )
    fade_in_seconds = max(0.0, float(subtitle_config.get("fade_in_ms", 0)) / 1000)
    fade_out_seconds = max(0.0, float(subtitle_config.get("fade_out_ms", 0)) / 1000)
    try:
        draft_font = getattr(draft.FontType, draft_font_name)
    except AttributeError as exc:
        raise ValueError(f"未知剪映字体：{draft_font_name}") from exc

    class RichTextSegment(draft.TextSegment):
        def __init__(self, *args: Any, highlight_range: tuple[int, int] | None = None, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.highlight_range = highlight_range

        def export_material(self) -> dict[str, Any]:
            material = super().export_material()
            if not self.highlight_range:
                return material
            content = json.loads(material["content"])
            base_style = content["styles"][0]
            start, end = self.highlight_range
            ranges: list[dict[str, Any]] = []
            for left, right, color in (
                (0, start, base_color),
                (start, end, highlight_color),
                (end, len(self.text), base_color),
            ):
                if right <= left:
                    continue
                style = copy.deepcopy(base_style)
                style["range"] = [left, right]
                style["fill"]["content"]["solid"]["color"] = list(color)
                ranges.append(style)
            content["styles"] = ranges
            material["content"] = json.dumps(content, ensure_ascii=False)
            return material

    first_segment = True
    for cue in transcript.get("cues", []):
        for start, end, highlight_range in _pieces(cue):
            start_units = round(start * draft.SEC)
            end_units = round(end * draft.SEC)
            if end_units <= start_units:
                continue
            segment = RichTextSegment(
                cue["text"],
                draft.Timerange(start_units, end_units - start_units),
                font=draft_font,
                style=draft.TextStyle(
                    size=draft_size,
                    bold=bool(subtitle_config.get("bold", True)),
                    color=base_color,
                    letter_spacing=int(
                        subtitle_config.get("draft_letter_spacing", 0)
                    ),
                    auto_wrapping=True,
                    max_line_width=float(subtitle_config.get("max_line_width", 0.86)),
                ),
                border=draft.TextBorder(
                    alpha=1.0, color=outline_color, width=border_width
                ),
                shadow=draft.TextShadow(
                    alpha=float(subtitle_config.get("draft_shadow_alpha", 0.45)),
                    color=shadow_color,
                    diffuse=float(subtitle_config.get("draft_shadow_diffuse", 18.0)),
                    distance=float(subtitle_config.get("draft_shadow_distance", 3.0)),
                    angle=float(subtitle_config.get("draft_shadow_angle", -45.0)),
                ),
                clip_settings=draft.ClipSettings(transform_y=transform_y),
                highlight_range=highlight_range,
            )
            if first_segment and style_preset == "history_hook":
                segment.add_animation(draft.TextIntro.轻微放大, duration=0.14)
            if abs(start - float(cue["start"])) < 0.001 and fade_in_seconds:
                segment.add_animation(
                    draft.TextIntro.渐显,
                    duration=min(fade_in_seconds, max(0.01, end - start)),
                )
            if abs(end - float(cue["end"])) < 0.001 and fade_out_seconds:
                segment.add_animation(
                    draft.TextOutro.渐隐,
                    duration=min(fade_out_seconds, max(0.01, end - start)),
                )
            script.add_segment(segment, track)
            first_segment = False
    if disclosure and disclosure.get("required"):
        disclosure_track = script.append_track(
            draft.TrackSpec(draft.TrackType.text, "ai_disclosure")
        )
        disclosure_end = float(disclosure["end"])
        disclosure_start = max(
            0.0, disclosure_end - float(disclosure.get("seconds", 2.0))
        )
        segment = draft.TextSegment(
            str(disclosure.get("text", "部分画面为 AI 历史重构")),
            draft.Timerange(
                round(disclosure_start * draft.SEC),
                round((disclosure_end - disclosure_start) * draft.SEC),
            ),
            font=draft.FontType.SourceHanSansCN_Bold,
            style=draft.TextStyle(
                size=max(7.5, draft_size - 1.5),
                bold=True,
                color=base_color,
                auto_wrapping=True,
                max_line_width=float(subtitle_config.get("max_line_width", 0.86)),
            ),
            border=draft.TextBorder(
                alpha=1.0, color=(0.03, 0.03, 0.03), width=border_width
            ),
            shadow=draft.TextShadow(
                alpha=0.45,
                color=(0.0, 0.0, 0.0),
                diffuse=18.0,
                distance=3.0,
                angle=-45.0,
            ),
            clip_settings=draft.ClipSettings(transform_y=-0.68),
        )
        script.add_segment(segment, disclosure_track)
    return track
