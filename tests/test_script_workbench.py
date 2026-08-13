from __future__ import annotations

import json
import http.client
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

from app import script_workbench as sw


class ScriptWorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "projects").mkdir()
        (self.root / "config").mkdir()
        (self.root / ".cache").mkdir()
        source_template = Path(__file__).resolve().parents[1] / "config" / "project_template.yaml"
        self.template = self.root / "config" / "project_template.yaml"
        self.template.write_text(source_template.read_text(encoding="utf-8"), encoding="utf-8")
        source_profiles = Path(__file__).resolve().parents[1] / "config" / "voice_profiles.yaml"
        self.voice_profiles = self.root / "config" / "voice_profiles.yaml"
        self.voice_profiles.write_text(
            source_profiles.read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.patches = [
            mock.patch.object(sw, "ROOT", self.root),
            mock.patch.object(sw, "DRAFTS_ROOT", self.root / "projects" / "_drafts"),
            mock.patch.object(sw, "TEMPLATE_PATH", self.template),
            mock.patch.object(sw, "VOICE_PROFILES_PATH", self.voice_profiles),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def _values(self, **overrides: str) -> dict[str, list[str]]:
        defaults = {
            "mode": "direct",
            "title": "测试历史故事",
            "duration_seconds": "45",
            "voice_profile": "yunyang_soft",
            "visual_strategy": "museum_and_ai",
            "topic": "",
            "source_material": "",
            "must_include": "",
            "avoid": "",
            "original_script": "这是我亲自写好的脚本。",
            "final_script": "这是我亲自写好的脚本。",
            "emphasis": "亲自写好",
            "proper_nouns": "",
            "pronunciation": "",
        }
        defaults.update(overrides)
        return {key: [value] for key, value in defaults.items()}

    def test_direct_lock_preserves_script_and_never_calls_ai(self) -> None:
        data = sw.create_draft()
        requester = mock.Mock()
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "不会调用"):
            sw.run_ai_revision(data, requester=requester)
        requester.assert_not_called()
        exact = "  这是我亲自写好的脚本。\n\n第二段保留空行。  "
        project = sw.lock_draft(
            data,
            self._values(original_script=exact, final_script=exact),
        )
        self.assertEqual((project / "script.txt").read_text(encoding="utf-8"), exact)
        config = yaml.safe_load((project / "project.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["subtitles"]["emphasis"]["include"], ["亲自写好"])
        self.assertEqual(config["voice"]["pronunciation"], {})
        self.assertEqual(config["voice"]["profile"], "yunyang_soft")
        self.assertNotIn("韦斯巴芗", (project / "project.yaml").read_text(encoding="utf-8"))

    def test_review_keeps_original_and_saves_structured_version(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(data, self._values(mode="review", final_script=""))
        requester = mock.Mock(
            return_value=(
                {
                    "suggested_script": "更有吸引力的建议稿。",
                    "summary": "开头需要更直接。",
                    "issues": [{"category": "钩子", "message": "偏慢", "suggestion": "直接进入反常识"}],
                    "risks": [{"type": "uncertain_fact", "message": "数字尚未核实"}],
                    "suggestions": {"emphasis": ["反常识"], "proper_nouns": ["古罗马"], "pronunciation": {}},
                },
                {"model": "deepseek-test"},
            )
        )
        with mock.patch.object(sw, "_ai_settings", return_value=({"model": "deepseek-test"}, "secret")):
            sw.run_ai_revision(data, requester=requester)
        self.assertEqual(data["original_script"], "这是我亲自写好的脚本。")
        self.assertEqual(data["final_script"], "更有吸引力的建议稿。")
        self.assertEqual(len(data["versions"]), 1)
        saved = json.loads(sw.draft_path(data["draft_id"]).read_text(encoding="utf-8"))
        self.assertNotIn("secret", json.dumps(saved, ensure_ascii=False))

    def test_review_api_failure_still_allows_locking_original(self) -> None:
        data = sw.create_draft()
        values = self._values(mode="review", final_script="")
        sw.update_from_form(data, values)
        with mock.patch.object(sw, "_ai_settings", return_value=({"model": "deepseek-test"}, "")):
            with self.assertRaisesRegex(sw.ScriptWorkbenchError, "DEEPSEEK_API_KEY"):
                sw.run_ai_revision(data)
        project = sw.lock_draft(data, values)
        self.assertEqual(
            (project / "script.txt").read_text(encoding="utf-8"),
            "这是我亲自写好的脚本。",
        )

    def test_topic_accepts_title_only_and_cached_result_skips_second_call(self) -> None:
        response = (
            {
                "suggested_script": "古代有一种令人意外的税。",
                "summary": "根据选题生成的初稿。",
                "issues": [],
                "risks": [{"type": "uncertain_fact", "message": "具体年代需要核实"}],
                "suggestions": {"emphasis": ["意外"], "proper_nouns": [], "pronunciation": {}},
            },
            {"model": "deepseek-test"},
        )
        first = sw.create_draft()
        values = self._values(
            mode="topic",
            topic="古代奇怪税收",
            source_material="",
            original_script="",
            final_script="",
        )
        sw.update_from_form(first, values)
        requester = mock.Mock(return_value=response)
        with mock.patch.object(sw, "_ai_settings", return_value=({"model": "deepseek-test"}, "secret")):
            sw.run_ai_revision(first, requester=requester)
        second = sw.create_draft()
        sw.update_from_form(second, values)
        with mock.patch.object(sw, "_ai_settings", return_value=({"model": "deepseek-test"}, "secret")):
            cached = sw.run_ai_revision(second, requester=mock.Mock(side_effect=AssertionError("cache miss")))
        self.assertEqual(requester.call_count, 1)
        self.assertTrue(cached["cache_hit"])
        self.assertTrue(second["analysis"]["risks"])
        project = sw.lock_draft(second, {**values, "final_script": [second["final_script"]]})
        self.assertTrue((project / "script.txt").is_file())

    def test_topic_failure_does_not_create_project(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(
            data,
            self._values(mode="topic", topic="古代奇怪税收", original_script="", final_script=""),
        )
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "尚未成功生成"):
            sw.lock_draft(data, self._values(mode="topic", topic="古代奇怪税收", original_script="", final_script=""))
        self.assertEqual([path.name for path in (self.root / "projects").iterdir()], ["_drafts"])

    def test_switching_review_draft_to_topic_requires_a_topic_generation(self) -> None:
        data = sw.create_draft()
        data["versions"] = [{"version": 1, "mode": "review", "script": "旧审稿"}]
        values = self._values(
            mode="topic",
            topic="新选题",
            original_script="",
            final_script="旧审稿",
        )
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "尚未成功生成"):
            sw.lock_draft(data, values)

    def test_duration_is_required_and_warning_does_not_block_lock(self) -> None:
        data = sw.create_draft()
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "选择目标时长"):
            sw.lock_draft(data, self._values(duration_seconds=""))
        short = sw.script_stats("很短。", 90)
        self.assertTrue(short["warnings"])
        project = sw.lock_draft(data, self._values(duration_seconds="90"))
        self.assertTrue(project.is_dir())

    def test_versions_are_limited_and_can_be_restored_without_changing_original(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(data, self._values(mode="review"))
        data["versions"] = [{"version": number} for number in range(1, 11)]
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "10 次"):
            sw.run_ai_revision(data, requester=mock.Mock())
        self.assertEqual(data["original_script"], "这是我亲自写好的脚本。")

    def test_manual_edit_is_snapshotted_before_ai_rewrites_it(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(data, self._values(mode="review", final_script="人工改写稿"))
        requester = mock.Mock(
            return_value=(
                {
                    "suggested_script": "新的 AI 建议稿",
                    "summary": "",
                    "issues": [],
                    "risks": [],
                    "suggestions": {},
                },
                {"model": "deepseek-test"},
            )
        )
        with mock.patch.object(sw, "_ai_settings", return_value=({"model": "deepseek-test"}, "secret")):
            sw.run_ai_revision(data, requester=requester)
        self.assertEqual(data["manual_versions"][0]["script"], "人工改写稿")
        self.assertEqual(data["final_script"], "新的 AI 建议稿")

    def test_autosaved_optional_fields_survive_reload(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(
            data,
            self._values(
                emphasis="陶罐，撒尿",
                proper_nouns="古罗马",
                pronunciation="韦斯巴芗=韦斯巴香",
            ),
        )
        restored = sw.load_draft(data["draft_id"])
        self.assertEqual(restored["suggestions"]["emphasis"], ["陶罐", "撒尿"])
        self.assertEqual(restored["suggestions"]["pronunciation"], {"韦斯巴芗": "韦斯巴香"})

    def test_voice_profile_is_autosaved_and_unknown_profile_is_rejected(self) -> None:
        data = sw.create_draft()
        sw.update_from_form(data, self._values(voice_profile="yunjian_story"))
        self.assertEqual(
            sw.load_draft(data["draft_id"])["voice_profile"], "yunjian_story"
        )
        with self.assertRaisesRegex(sw.ScriptWorkbenchError, "profile"):
            sw.update_from_form(data, self._values(voice_profile="../../secret"))

    def test_voice_form_reads_profiles_dynamically(self) -> None:
        data = sw.create_draft()
        page = sw._form(data, "test-csrf")
        self.assertIn('name="voice_profile"', page)
        self.assertIn("云扬舒缓叙事", page)
        self.assertIn("zh-CN-YunyangNeural", page)

    def test_lock_creates_unique_projects_and_manifest_without_secrets(self) -> None:
        values = self._values(
            pronunciation="陶罐=陶罐",
            visual_strategy="ai_only",
            voice_profile="yunjian_story",
        )
        with mock.patch.object(sw, "_project_id", side_effect=["history-a", "history-b"]):
            first = sw.create_draft()
            first_dir = sw.lock_draft(first, values)
            second = sw.create_draft()
            second_dir = sw.lock_draft(second, values)
        self.assertNotEqual(first_dir, second_dir)
        manifest = json.loads((first_dir / "script_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["publish_ready"])
        self.assertEqual(manifest["visual_strategy"], "ai_only")
        self.assertEqual(manifest["voice_profile"], "yunjian_story")
        config = yaml.safe_load((first_dir / "project.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["visuals"]["strategy"], "ai_only")
        self.assertEqual(config["voice"]["profile"], "yunjian_story")
        self.assertEqual(config["visuals"]["ai_fallback"]["candidates_per_shot"], 4)
        self.assertNotIn("API_KEY", json.dumps(manifest))
        self.assertTrue((first_dir / "script_history" / "final-edit.json").is_file())

    def test_local_server_escapes_html_checks_csrf_limits_body_and_shuts_down(self) -> None:
        draft = sw.create_draft()
        draft["title"] = '<script>alert("x")</script>'
        sw.save_draft(draft)
        output = self.root / "outputs" / "failed-task"
        output.mkdir(parents=True)
        (self.root / "projects" / "history-test").mkdir()
        (output / "task.json").write_text(
            json.dumps(
                {
                    "task_id": "failed-task",
                    "project_id": "history-test",
                    "status": "failed",
                    "current_stage": "preview",
                    "options": {"visual_mode": "sourced"},
                }
            ),
            encoding="utf-8",
        )
        audition = self.root / "outputs" / "voice-audition-test"
        audition.mkdir()
        preview_bytes = b"ID3-test-audio"
        (audition / "04_yunyang_soft.mp3").write_bytes(preview_bytes)
        (audition / "comparison.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "profile": "yunyang_soft",
                            "file": "04_yunyang_soft.mp3",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        ready = threading.Event()
        address: list[str] = []
        result: list[dict[str, object]] = []

        def run() -> None:
            result.append(
                sw.run_workbench_server(
                    open_browser=False,
                    ready_callback=lambda url: (address.append(url), ready.set()),
                )
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(3))
        response = urllib.request.urlopen(address[0], timeout=3)
        page = response.read().decode("utf-8")
        self.assertEqual(response.headers.get("Content-Security-Policy").split(";")[0], "default-src 'self'")
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn('<script>alert("x")</script>', page)
        preview_response = urllib.request.urlopen(
            address[0] + "voice-preview/yunyang_soft", timeout=3
        )
        self.assertEqual(preview_response.headers.get_content_type(), "audio/mpeg")
        self.assertEqual(preview_response.read(), preview_bytes)
        with self.assertRaises(urllib.error.HTTPError) as missing_preview:
            urllib.request.urlopen(address[0] + "voice-preview/not_allowed", timeout=3)
        self.assertEqual(missing_preview.exception.code, 404)
        token = re.search(r'name=csrf value="([^"]+)"', page).group(1)  # type: ignore[union-attr]
        bad = urllib.request.Request(
            address[0] + "home",
            data=urllib.parse.urlencode({"csrf": "wrong", "action": "resume_video"}).encode(),
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(bad, timeout=3)
        self.assertEqual(context.exception.code, 403)
        parsed = urllib.parse.urlsplit(address[0])
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
        connection.request(
            "POST",
            "/home",
            body=b"",
            headers={"Content-Length": str(sw.MAX_REQUEST_BYTES + 1)},
        )
        self.assertEqual(connection.getresponse().status, 413)
        connection.close()
        good = urllib.request.Request(
            address[0] + "home",
            data=urllib.parse.urlencode({"csrf": token, "action": "resume_video"}).encode(),
            method="POST",
        )
        urllib.request.urlopen(good, timeout=3).read()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "resume_video")


if __name__ == "__main__":
    unittest.main()
