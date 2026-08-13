from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from app.jianying_text import add_rich_subtitles


class _Value:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


class _Timerange:
    def __init__(self, start: int, duration: int):
        self.start = start
        self.duration = duration


class _TextSegment:
    def __init__(self, text: str, timerange: _Timerange, **kwargs: Any):
        self.text = text
        self.target_timerange = timerange
        self.kwargs = kwargs
        self.animations: list[tuple[Any, Any]] = []

    def add_animation(self, *args: Any, **kwargs: Any) -> None:
        self.animations.append((args, kwargs))

    def export_material(self) -> dict[str, Any]:
        return {"content": '{"styles": [{}]}'}


class _TrackSpec:
    def __init__(self, track_type: str, name: str):
        self.track_type = track_type
        self.name = name


class _Script:
    def __init__(self):
        self.tracks: list[dict[str, Any]] = []

    def append_track(self, spec: _TrackSpec) -> dict[str, Any]:
        track = {"name": spec.name, "segments": []}
        self.tracks.append(track)
        return track

    def add_segment(self, segment: _TextSegment, track: dict[str, Any]) -> None:
        start = segment.target_timerange.start
        end = start + segment.target_timerange.duration
        for existing in track["segments"]:
            existing_start = existing.target_timerange.start
            existing_end = existing_start + existing.target_timerange.duration
            if start < existing_end and existing_start < end:
                raise ValueError("segments overlap on one track")
        track["segments"].append(segment)


_DRAFT = SimpleNamespace(
    SEC=1_000_000,
    TrackType=SimpleNamespace(text="text"),
    TrackSpec=_TrackSpec,
    TextSegment=_TextSegment,
    Timerange=_Timerange,
    FontType=SimpleNamespace(SourceHanSansCN_Bold="source-han-sans-bold"),
    TextStyle=_Value,
    TextBorder=_Value,
    TextShadow=_Value,
    ClipSettings=_Value,
    TextIntro=SimpleNamespace(渐显="fade-in"),
    TextOutro=SimpleNamespace(渐隐="fade-out"),
)


class JianyingTextTrackTests(unittest.TestCase):
    def test_caption_adds_one_fade_in_and_out_animation(self):
        script = _Script()

        add_rich_subtitles(
            script,
            _DRAFT,
            {"cues": [{"start": 0.0, "end": 1.0, "text": "测试字幕"}]},
            {
                "style_preset": "social_pink",
                "draft_font_type": "SourceHanSansCN_Bold",
                "fade_in_ms": 150,
                "fade_out_ms": 150,
            },
        )

        animations = script.tracks[0]["segments"][0].animations
        self.assertEqual([item[0][0] for item in animations], ["fade-in", "fade-out"])
        self.assertEqual([item[1]["duration"] for item in animations], [0.15, 0.15])

    def test_ai_disclosure_uses_a_separate_track_when_it_overlaps_last_caption(self):
        script = _Script()
        transcript = {
            "cues": [
                {
                    "start": 47.50,
                    "end": 49.03,
                    "text": "钱，是没有味道的。",
                }
            ]
        }
        disclosure = {
            "required": True,
            "text": "部分画面为 AI 历史重构",
            "seconds": 2.0,
            "end": 49.92,
        }

        add_rich_subtitles(
            script,
            _DRAFT,
            transcript,
            {"style_preset": "history_keyword"},
            disclosure=disclosure,
        )

        self.assertEqual([track["name"] for track in script.tracks], [
            "subtitles",
            "ai_disclosure",
        ])
        self.assertEqual(len(script.tracks[0]["segments"]), 1)
        self.assertEqual(len(script.tracks[1]["segments"]), 1)
        caption = script.tracks[0]["segments"][0].target_timerange
        notice = script.tracks[1]["segments"][0].target_timerange
        self.assertLess(notice.start, caption.start + caption.duration)
        self.assertEqual(notice.start + notice.duration, 49_920_000)


if __name__ == "__main__":
    unittest.main()
