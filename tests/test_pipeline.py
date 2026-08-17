from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.pipeline import (
    BuildError,
    Cue,
    _zoompan_filter,
    build_project,
    merge_cues,
    parse_edge_metadata,
    parse_srt,
    schedule_shots,
    whisper_initial_prompt,
    wrap_caption,
    write_ass,
)
from app.task_state import TaskState, find_task_dir
from app.transcript import AlignmentQualityError, build_transcript


class MotionFilterTests(unittest.TestCase):
    def test_pan_uses_cosine_easing_and_one_multi_frame_zoompan(self) -> None:
        result = _zoompan_filter(
            "pan_right",
            105,
            1080,
            1920,
            30,
            zoom_amount=0.035,
            easing="cosine",
        )

        self.assertIn("cos(PI*on/104)", result)
        self.assertIn("d=105", result)
        self.assertIn("fps=30", result)
        self.assertIn("1.035000", result)
        self.assertNotIn("d=1:", result)

    def test_static_motion_does_not_pan_or_zoom(self) -> None:
        result = _zoompan_filter("static", 60, 1080, 1920, 30)

        self.assertIn("z='1.0'", result)
        self.assertIn("x='0'", result)


class WhisperBuildBranchTests(unittest.TestCase):
    def test_new_whisper_alignment_uses_resolved_voice_config(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project_dir = root / "project"
            output_root = root / "outputs"
            raw_dir = root / "raw"
            draft_root = root / "drafts"
            for path in (project_dir, output_root, raw_dir, draft_root):
                path.mkdir()
            script_path = project_dir / "script.txt"
            script_path.write_text("化学家舍勒发现了这种颜料。", encoding="utf-8")
            config = {
                "version": 1,
                "project": {"id": "whisper-branch-test", "script_file": "script.txt"},
                "voice": {
                    "voice": "zh-CN-YunyangNeural",
                    "pronunciation": {"舍勒": "Shè lè"},
                },
                "subtitles": {
                    "style_preset": "history_clean",
                    "alignment_provider": "moneyprinter_whisper",
                },
                "visuals": {"mode": "local"},
            }
            resolved_voice = {
                **config["voice"],
                "provider": "moneyprinter_edge",
                "profile": "snapshot",
                "label": "zh-CN-YunyangNeural",
                "rate": "+0%",
                "pitch": "+0Hz",
                "pauses": {},
            }
            observed_prompt: list[str] = []

            def fake_tts(text: str, voice: dict, run_dir: Path):
                raw = run_dir / "narration.raw.mp3"
                normalized = run_dir / "narration.mp3"
                metadata = run_dir / "captions_metadata.jsonl"
                raw.write_bytes(b"test")
                normalized.write_bytes(b"test")
                metadata.write_text("{}\n", encoding="utf-8")
                return raw, normalized, metadata, False, resolved_voice

            def fake_alignment(
                narration: Path,
                subtitle_config: dict,
                run_dir: Path,
                *,
                initial_prompt: str = "",
            ):
                observed_prompt.append(initial_prompt)
                path = run_dir / "working" / "alignment.json"
                payload = {"words": []}
                path.write_text(json.dumps(payload), encoding="utf-8")
                return payload, path, False

            with (
                patch("app.pipeline.configure_pipeline_paths"),
                patch("app.pipeline._resolve_project", return_value=project_dir),
                patch("app.pipeline._load_config", return_value=config),
                patch(
                    "app.pipeline._preflight",
                    return_value=(script_path, raw_dir, draft_root),
                ),
                patch("app.pipeline._build_input_hash", return_value="test-hash"),
                patch(
                    "app.pipeline.get_studio_paths",
                    return_value=SimpleNamespace(output_root=output_root),
                ),
                patch("app.pipeline.create_or_reuse_tts", side_effect=fake_tts),
                patch("app.pipeline.probe_audio_duration", return_value=2.0),
                patch(
                    "app.pipeline.create_or_reuse_alignment",
                    side_effect=fake_alignment,
                ),
                patch(
                    "app.pipeline.build_transcript",
                    side_effect=BuildError("stop-after-alignment"),
                ),
            ):
                with self.assertRaisesRegex(BuildError, "stop-after-alignment"):
                    build_project("ignored", skip_draft=True)

            self.assertEqual(observed_prompt, ["舍勒"])


class SubtitleTests(unittest.TestCase):
    def test_parse_edge_metadata_uses_100ns_ticks(self) -> None:
        item = {
            "type": "SentenceBoundary",
            "offset": 1_000_000,
            "duration": 20_000_000,
            "text": "测试字幕。",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.jsonl"
            path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
            cues = parse_edge_metadata(path)
        self.assertEqual(cues, [Cue(0.1, 2.1, "测试字幕。")])

    def test_parse_srt_preserves_two_lines(self) -> None:
        content = "1\n00:00:00,000 --> 00:00:02,000\n第一行字幕\n第二行字幕\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "captions.srt"
            path.write_text(content, encoding="utf-8")
            cues = parse_srt(path)
        self.assertEqual(cues[0].text, "第一行字幕\n第二行字幕")

    def test_wrap_caption_never_exceeds_two_lines(self) -> None:
        wrapped = wrap_caption("古罗马人会把衣服放进混着尿液的水槽", 14, 2)
        lines = wrapped.splitlines()
        self.assertLessEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 14 for line in lines))

    def test_wrap_caption_does_not_leave_a_single_character_orphan(self) -> None:
        wrapped = wrap_caption("它其实是让路人往里面撒尿的。", 12, 2)
        lines = wrapped.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertGreaterEqual(min(len(line) for line in lines), 4)

    def test_merge_cues_is_monotonic(self) -> None:
        cues = [
            Cue(0.0, 0.5, "古罗马"),
            Cue(0.5, 1.0, "人洗衣服，"),
            Cue(1.0, 1.6, "真的会用尿。"),
        ]
        merged = merge_cues(cues, 14, 2)
        self.assertGreaterEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 0.0)
        self.assertTrue(all(a.end <= b.start for a, b in zip(merged, merged[1:])))


class SchedulingTests(unittest.TestCase):
    def test_schedule_is_frame_aligned_and_complete(self) -> None:
        shots = schedule_shots(
            duration=58.413,
            fps=30,
            hook_until=5.0,
            hook_range=(1.5, 2.0),
            body_range=(3.0, 4.0),
            seed=20260810,
        )
        self.assertGreaterEqual(sum(1 for shot in shots if shot["start"] < 5.0), 3)
        self.assertAlmostEqual(shots[-1]["end"], round(58.413 * 30) / 30, places=6)
        self.assertTrue(all(abs(shot["duration"] * 30 - round(shot["duration"] * 30)) < 1e-6 for shot in shots))


class TranscriptTests(unittest.TestCase):
    def _alignment(self, text: str) -> dict:
        semantic = [char for char in text if char.strip() and char not in "，。！？"]
        return {
            "engine": "test-whisper",
            "model": "tiny-test",
            "device": "cpu",
            "words": [
                {
                    "text": char,
                    "start": index * 0.11,
                    "end": (index + 1) * 0.11,
                    "probability": 0.99,
                }
                for index, char in enumerate(semantic)
            ],
        }

    def _recognized_alignment(self, text: str) -> dict:
        semantic = [char for char in text if char.isalnum() or "\u3400" <= char <= "\u9fff"]
        return {
            "engine": "test-whisper",
            "model": "tiny-test",
            "device": "cpu",
            "words": [
                {
                    "text": char,
                    "start": index * 0.11,
                    "end": (index + 1) * 0.11,
                    "probability": 0.99,
                }
                for index, char in enumerate(semantic)
            ],
        }

    def test_transcript_reflows_and_highlights_manual_phrase(self) -> None:
        text = "如果来到古罗马，街角的大陶罐其实是让路人撒尿的。更离谱的是它还能赚钱。"
        config = {
            "max_chars_per_line": 12,
            "max_lines": 2,
            "target_chars_per_cue": 12,
            "min_cue_seconds": 0.8,
            "max_cue_seconds": 2.8,
            "minimum_alignment_coverage": 0.98,
            "emphasis": {"mode": "hybrid", "include": ["大陶罐", "撒尿"]},
        }
        transcript, cues, emphasis = build_transcript(
            text,
            self._alignment(text),
            audio_duration=8.0,
            subtitle_config=config,
        )
        self.assertGreaterEqual(transcript["alignment"]["coverage"], 0.98)
        self.assertTrue(all(cue.end - cue.start <= 2.82 for cue in cues))
        self.assertTrue(all(len(line) <= 12 for cue in cues for line in cue.text.splitlines()))
        self.assertTrue(any(item["phrase"] == "大陶罐" for item in emphasis))

    def test_low_alignment_coverage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "覆盖率"):
            build_transcript(
                "古罗马人使用大陶罐",
                {
                    "words": [
                        {"text": "完全不同", "start": 0.0, "end": 1.0, "probability": 0.5}
                    ]
                },
                audio_duration=2.0,
                subtitle_config={
                    "minimum_alignment_coverage": 0.98,
                    "emphasis": {"mode": "hybrid"},
                },
            )

    def test_homophone_replacements_are_accepted_and_recorded(self) -> None:
        text = (
            "一名19岁的女工，死因是砷中毒。她的工作，竟是给帽子做绿叶。"
            "这绿鲜艳又便宜。舍勒发明了这种含砷颜料，粉尘侵入肺腑，她们是牺牲品。"
        )
        recognized = (
            "一名19岁的女工死因弑身中毒她的工作竟是给帽子做滤液"
            "这滤鲜艳又便宜舍勒发明了这种寒深颜料粉尘亲入肺腑他们是牺牲品"
        )
        transcript, cues, _ = build_transcript(
            text,
            self._recognized_alignment(recognized),
            audio_duration=12.0,
            subtitle_config={
                "minimum_alignment_coverage": 0.98,
                "maximum_missing_ratio": 0.01,
                "maximum_unresolved_run_chars": 3,
                "phonetic_matching": True,
                "max_phonetic_span_chars": 4,
                "emphasis": {"mode": "none"},
            },
        )
        alignment = transcript["alignment"]
        self.assertLess(alignment["exact_coverage"], 0.98)
        self.assertEqual(alignment["phonetic_coverage"], 1.0)
        self.assertEqual(alignment["missing_ratio"], 0.0)
        self.assertEqual(alignment["result"], "passed")
        accepted = [item for item in alignment["mismatches"] if item["accepted"]]
        self.assertEqual(len(accepted), 6)
        self.assertTrue(all(item["classification"] == "phonetic_match" for item in accepted))
        self.assertEqual("".join(cue.text.replace("\n", "") for cue in cues), text)

    def test_non_homophone_and_numeric_replacements_are_rejected(self) -> None:
        for script, recognized in (
            ("绿色颜料很危险", "红色颜料很危险"),
            ("一名19岁的女工", "一名18岁的女工"),
            ("她她她她她来到工厂", "他他他他他来到工厂"),
        ):
            with self.subTest(script=script), self.assertRaises(AlignmentQualityError) as raised:
                build_transcript(
                    script,
                    self._recognized_alignment(recognized),
                    audio_duration=3.0,
                    subtitle_config={
                        "minimum_alignment_coverage": 0.98,
                        "phonetic_matching": True,
                        "max_phonetic_span_chars": 4,
                        "emphasis": {"mode": "none"},
                    },
                )
            self.assertEqual(raised.exception.diagnostics["result"], "failed")

    def test_long_missing_and_unexpected_phrases_are_rejected(self) -> None:
        cases = (
            ("古罗马人使用绿色颜料", "古罗马人使用"),
            ("古罗马人使用绿色颜料", "古罗马人真的很危险使用绿色颜料"),
        )
        for script, recognized in cases:
            with self.subTest(recognized=recognized), self.assertRaises(AlignmentQualityError):
                build_transcript(
                    script,
                    self._recognized_alignment(recognized),
                    audio_duration=3.0,
                    subtitle_config={
                        "minimum_alignment_coverage": 0.98,
                        "maximum_missing_ratio": 0.01,
                        "maximum_unresolved_run_chars": 3,
                        "emphasis": {"mode": "none"},
                    },
                )

    def test_empty_whisper_result_returns_writable_diagnostics(self) -> None:
        with self.assertRaises(AlignmentQualityError) as raised:
            build_transcript(
                "古罗马人使用大陶罐",
                {"words": []},
                audio_duration=2.0,
                subtitle_config={
                    "minimum_alignment_coverage": 0.98,
                    "emphasis": {"mode": "none"},
                },
            )
        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["result"], "failed")
        self.assertEqual(diagnostics["missing_ratio"], 1.0)
        self.assertEqual(diagnostics["mismatches"][0]["classification"], "missing")

    def test_whisper_prompt_only_contains_terms_present_in_script(self) -> None:
        prompt = whisper_initial_prompt(
            "瑞典化学家舍勒发明了含砷颜料。",
            {"pronunciation": {"舍勒": "Shè lè", "玛蒂尔达": "Mǎ dì ěr dá"}},
            {"emphasis": {"proper_nouns": ["含砷颜料", "古罗马"]}},
        )
        self.assertEqual(prompt, "舍勒、含砷颜料")
        self.assertNotIn("瑞典化学家", prompt)

    def test_chinese_words_do_not_create_short_single_character_tails(self) -> None:
        text = "于是对公共厕所收集的尿液征税。他就拿起一枚金币放到儿子鼻子前，问：有味道吗？"
        config = {
            "max_chars_per_line": 12,
            "max_lines": 2,
            "target_chars_per_cue": 16,
            "min_cue_seconds": 0.8,
            "max_cue_seconds": 2.8,
            "minimum_alignment_coverage": 0.98,
            "emphasis": {"mode": "hybrid", "include": ["尿液征税"]},
        }
        transcript, cues, _ = build_transcript(
            text,
            self._alignment(text),
            audio_duration=5.0,
            subtitle_config=config,
        )
        flat_cues = [cue.text.replace("\n", "") for cue in cues]
        self.assertTrue(all(cue.end - cue.start >= 0.78 for cue in cues))
        self.assertFalse(any(left.endswith("征") and right.startswith("税") for left, right in zip(flat_cues, flat_cues[1:])))
        self.assertFalse(any(left.endswith("鼻") and right.startswith("子") for left, right in zip(flat_cues, flat_cues[1:])))
        self.assertEqual(transcript["alignment"]["coverage"], 1.0)

    def test_single_line_cards_are_at_most_eight_chars_and_keep_short_phrases(self) -> None:
        text = "更离谱的是，洗衣店就会派人来收。"
        transcript, cues, _ = build_transcript(
            text,
            self._alignment(text),
            audio_duration=3.0,
            subtitle_config={
                "max_chars_per_line": 8,
                "max_lines": 1,
                "target_chars_per_cue": 6,
                "min_cue_seconds": 0.3,
                "max_cue_seconds": 2.0,
                "minimum_alignment_coverage": 0.98,
                "preserve_short_punctuation_cues": True,
                "minimum_short_cue_chars": 2,
                "emphasis": {"mode": "none"},
            },
        )

        self.assertTrue(all("\n" not in cue.text and len(cue.text) <= 8 for cue in cues))
        self.assertTrue(any(cue.text.startswith("洗衣店") for cue in cues))
        self.assertTrue(any(len(cue.text.rstrip("，。！？：；")) <= 4 for cue in cues))
        self.assertFalse(any(cue.text.startswith(("，", "。", "：")) for cue in cues))
        self.assertEqual(transcript["limits"]["max_lines"], 1)

    def test_ass_contains_timed_keyword_overlay(self) -> None:
        cues = [Cue(0.0, 2.0, "古罗马\n大陶罐")]
        transcript = {
            "cues": [
                {
                    "text": "古罗马\n大陶罐",
                    "emphasis": {
                        "text_start": 3,
                        "text_end": 6,
                        "start": 0.8,
                        "end": 1.4,
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "captions.ass"
            write_ass(
                path,
                cues,
                {"width": 1080, "height": 1920},
                {
                    "font_name": "Source Han Sans SC Heavy",
                    "font_size": 76,
                    "base_color": "#FFFFFF",
                    "highlight_color": "#FFD54A",
                    "outline": 5,
                    "shadow": 2,
                    "margin_bottom": 340,
                    "style_preset": "history_keyword",
                },
                transcript,
            )
            content = path.read_text(encoding="utf-8-sig")
        self.assertIn("Dialogue: 1,0:00:00.80,0:00:01.40", content)
        self.assertIn("大陶罐", content)
        self.assertIn("alpha&HFF", content)
        self.assertIn(r"\1c&H4AD5FF&}", content)

    def test_ass_supports_pink_fill_outline_shadow_and_non_bold_font(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "captions.ass"
            write_ass(
                path,
                [Cue(0.0, 2.0, "在罗马尼亚的新年")],
                {"width": 1080, "height": 1920},
                {
                    "font_name": "FZShuTi",
                    "font_size": 108,
                    "base_color": "#FFE3EC",
                    "highlight_color": "#FFF3B0",
                    "outline_color": "#FF5C91",
                    "shadow_color": "#851F49",
                    "shadow_opacity": 0.82,
                    "bold": False,
                    "letter_spacing": 2,
                    "outline": 8,
                    "shadow": 5,
                    "margin_bottom": 460,
                    "style_preset": "social_pink",
                    "fade_in_ms": 150,
                    "fade_out_ms": 150,
                },
            )
            content = path.read_text(encoding="utf-8-sig")

        self.assertIn("FZShuTi,108", content)
        self.assertIn("&H00ECE3FF", content)
        self.assertIn("&H00915CFF", content)
        self.assertIn("&H2E491F85", content)
        self.assertIn(",0,0,0,0,100,100,2.0,0,1,8,5,2,70,70,460,1", content)
        self.assertIn(r"{\fad(150,150)}", content)


class TaskStateTests(unittest.TestCase):
    def test_failed_task_can_be_found_and_stage_reused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder)
            run_dir = outputs / "roman-20260811"
            run_dir.mkdir()
            artifact = run_dir / "narration.mp3"
            artifact.write_bytes(b"audio")
            task = TaskState.create(
                run_dir / "task.json",
                task_id=run_dir.name,
                run_id="20260811",
                project_id="roman",
                input_hash="same-input",
            )
            task.begin("voice", "same-input")
            task.succeed("voice", "same-input", [artifact])
            task.fail("draft", "剪映正在运行")
            self.assertTrue(task.can_reuse("voice", "same-input"))
            self.assertEqual(find_task_dir(outputs, "latest", project_id="roman"), run_dir)

    def test_latest_task_can_be_scoped_to_compatible_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder) / "outputs"
            outputs.mkdir()
            old_run = outputs / "roman-old"
            new_run = outputs / "roman-new"
            old_run.mkdir()
            new_run.mkdir()
            compatible = TaskState.create(
                old_run / "task.json",
                task_id=old_run.name,
                run_id="old",
                project_id="roman",
                input_hash="compatible",
            )
            compatible.fail("asset_search", "retry")
            incompatible = TaskState.create(
                new_run / "task.json",
                task_id=new_run.name,
                run_id="new",
                project_id="roman",
                input_hash="changed",
            )
            incompatible.fail("asset_search", "newer but incompatible")
            self.assertEqual(
                find_task_dir(
                    outputs,
                    "latest",
                    project_id="roman",
                    input_hash="compatible",
                ),
                old_run,
            )
            with self.assertRaises(FileNotFoundError):
                find_task_dir(
                    outputs,
                    "latest",
                    project_id="roman",
                    input_hash="missing",
                )

    def test_latest_task_is_scoped_to_the_same_draft_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder) / "outputs"
            official_root = Path(folder) / "official-drafts"
            test_root = Path(folder) / "test-drafts"
            outputs.mkdir()
            official = outputs / "roman-official"
            test = outputs / "roman-test"
            official.mkdir()
            test.mkdir()
            official_task = TaskState.create(
                official / "task.json",
                task_id=official.name,
                run_id="official",
                project_id="roman",
                input_hash="official-input",
                options={"draft_root": str(official_root)},
            )
            official_task.fail("draft", "剪映正在运行")
            test_task = TaskState.create(
                test / "task.json",
                task_id=test.name,
                run_id="test",
                project_id="roman",
                input_hash="test-input",
                options={"draft_root": str(test_root)},
            )
            test_task.fail("draft", "test failure")
            self.assertEqual(
                find_task_dir(
                    outputs,
                    "latest",
                    project_id="roman",
                    draft_root=official_root,
                ),
                official,
            )


if __name__ == "__main__":
    unittest.main()
