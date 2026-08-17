from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from app.asset_reuse import (
    AssetReuseError,
    _validate_reuse_selection,
    build_reuse_plan,
    create_reuse_source_snapshot,
    merge_reused_candidates,
    resolve_parent_task,
    unmatched_scene_plan,
)
from app.task_state import REUSE_TASK_STAGES, TaskState
from app.visual_supply import download_selected_assets
from app.pipeline import render_shot
from app import pipeline


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _image(path: Path, color: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1024, 1536), color).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intent(shot_id: int, narration: str, obj: str) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "intent_id": f"intent-{shot_id:03d}",
        "narration": narration,
        "must_include": [obj],
        "avoid": ["modern car"],
        "objects": [obj],
        "people": [],
        "search_terms_en": [obj],
        "search_terms_zh": [],
        "time_context": {
            "label": "Ancient Rome",
            "period_terms_en": ["Ancient Rome"],
        },
    }


class AssetReuseSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.outputs = root / "outputs"
        self.cache = root / "cache"
        self.run = self.outputs / "parent-task"
        self.run.mkdir(parents=True)
        (self.cache / "assets").mkdir(parents=True)
        (self.cache / "ai").mkdir(parents=True)
        museum_file = root / "museum.jpg"
        museum_sha = _image(museum_file, "red")
        (self.cache / "assets" / museum_sha).write_bytes(museum_file.read_bytes())
        ai_one = self.cache / "ai" / "ai-one.jpg"
        ai_two = self.cache / "ai" / "ai-two.jpg"
        ai_one_sha = _image(ai_one, "blue")
        ai_two_sha = _image(ai_two, "green")
        self.museum = {
            "asset_id": "met-selected",
            "provider": "met",
            "source_id": "1",
            "title": "Roman jar",
            "creator": "Unknown",
            "institution": "Met",
            "source_page": "https://example.test/met/1",
            "download_url": "https://example.test/met/1.jpg",
            "rights_code": "pdm-1.0",
            "rights_url": "https://creativecommons.org/publicdomain/mark/1.0/",
            "width": 1024,
            "height": 1536,
            "mime": "image/jpeg",
            "selectable": True,
            "ai_generated": False,
        }
        self.ai_one = {
            "asset_id": "comfyui-one",
            "provider": "comfyui_local",
            "source_id": "a",
            "title": "AI one",
            "creator": "ComfyUI",
            "institution": "Local",
            "source_page": "http://127.0.0.1:8000/",
            "download_url": "",
            "rights_code": "provider_terms",
            "rights_url": "https://example.test/model-license",
            "width": 1024,
            "height": 1536,
            "mime": "image/jpeg",
            "selectable": True,
            "ai_generated": True,
            "local_preview": str(ai_one),
            "generation": {"sha256": ai_one_sha, "prompt": "Roman jar"},
        }
        self.ai_two = {
            **self.ai_one,
            "asset_id": "comfyui-two",
            "source_id": "b",
            "title": "AI two",
            "local_preview": str(ai_two),
            "generation": {"sha256": ai_two_sha, "prompt": "Roman basin"},
        }
        unselected_museum = {**self.museum, "asset_id": "met-unselected", "source_id": "2"}
        _write_json(
            self.run / "task.json",
            {
                "schema_version": 1,
                "task_id": "parent-task",
                "project_id": "parent-project",
                "status": "succeeded",
                "input_hash": "parent-hash",
                "options": {"visual_mode": "sourced"},
            },
        )
        _write_json(self.run / "scene_plan.json", {"shots": [_intent(1, "陶罐放在街角", "Roman jar"), _intent(2, "工人踩洗衣物", "washing basin")]})
        _write_json(
            self.run / "asset_candidates.json",
            {
                "shots": [
                    {"shot_id": 1, "candidates": [self.museum, unselected_museum, self.ai_one]},
                    {"shot_id": 2, "candidates": [self.ai_two]},
                ]
            },
        )
        _write_json(
            self.run / "asset_selection.json",
            {"reviewed": True, "selections": [{"shot_id": 1, "asset_id": "met-selected", "candidate": self.museum}]},
        )
        _write_json(
            self.run / "assets_manifest.json",
            {
                "human_reviewed": True,
                "assets": [{"shot_id": 1, "asset_id": "met-selected", "sha256": museum_sha, "rights_code": "pdm-1.0"}],
            },
        )
        _write_json(self.run / "license_audit.json", {"asset_rights_ready": True})
        (self.run / "licenses.csv").write_text("shot_id,asset_id\n1,met-selected\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_snapshot_contains_selected_and_all_ai_but_not_unselected_museum(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        ids = {item["asset_id"] for item in snapshot["assets"]}
        self.assertEqual(ids, {"met-selected", "comfyui-one", "comfyui-two"})
        self.assertEqual(snapshot["selected_count"], 1)
        self.assertEqual(snapshot["unused_ai_count"], 2)
        self.assertNotIn(str(self.cache), json.dumps(snapshot, ensure_ascii=False))

    def test_parent_task_path_cannot_escape_output_root(self) -> None:
        with self.assertRaisesRegex(AssetReuseError, "ID 无效"):
            resolve_parent_task(self.outputs, "../parent-task")

    def test_changed_parent_cache_hash_fails_closed(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        museum = next(item for item in snapshot["assets"] if item["asset_id"] == "met-selected")
        (self.cache / "assets" / museum["sha256"]).write_bytes(b"changed")
        with self.assertRaisesRegex(AssetReuseError, "哈希"):
            create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")

    def test_exact_match_is_preselected_without_deepseek_and_fuzzy_is_manual(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        scene = {"shots": [_intent(1, "陶罐放在街角", "Roman jar"), _intent(2, "全新的一段旁白", "coin")]}
        plan = build_reuse_plan(
            scene,
            snapshot,
            {"recommendation_threshold": 75, "alternative_threshold": 20},
            {"model": "test", "secret_ref": "MISSING"},
            {},
            self.cache,
        )
        self.assertEqual(plan["semantic_status"], "unavailable")
        self.assertEqual(plan["shots"][0]["recommended_asset_id"], "met-selected")
        self.assertIsNone(plan["shots"][1]["recommended_asset_id"])

    def test_deepseek_matching_is_batched_and_successful_batches_are_cached(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        scene = {
            "shots": [
                _intent(index, f"重新改写的镜头{index}", f"object-{index}")
                for index in range(1, 6)
            ]
        }

        def response(_: str, payload: dict[str, object], __: dict[str, object], ___: str):
            judgments = []
            for row in payload["shots"]:  # type: ignore[index]
                for candidate in row["candidates"]:
                    judgments.append(
                        {
                            "shot_id": row["shot_id"],
                            "asset_id": candidate["asset_id"],
                            "score": 80,
                            "verdict": "recommend",
                            "reason": "metadata fits",
                            "conflicts": [],
                        }
                    )
            return {"judgments": judgments}, {"model": "mock"}

        settings = {"model": "mock", "secret_ref": "KEY"}
        config = {"recommendation_threshold": 75, "alternative_threshold": 20, "shots_per_batch": 2}
        with mock.patch("app.asset_reuse.request_json_object", side_effect=response) as requester:
            first = build_reuse_plan(scene, snapshot, config, settings, {"KEY": "secret"}, self.cache)
        self.assertEqual(requester.call_count, 3)
        self.assertEqual(first["semantic_status"], "reviewed")
        with mock.patch("app.asset_reuse.request_json_object", side_effect=AssertionError("cache miss")):
            second = build_reuse_plan(scene, snapshot, config, settings, {"KEY": "secret"}, self.cache)
        self.assertEqual(second["semantic_status"], "reviewed")
        self.assertTrue(all(batch.get("cache_hit") for batch in second["deepseek_response"]["batches"]))

    def test_duplicate_requires_explicit_override(self) -> None:
        plan = {
            "shots": [
                {"shot_id": 1, "candidates": [{"asset_id": "same", "score": 100, "reason": "exact", "conflicts": []}]},
                {"shot_id": 2, "candidates": [{"asset_id": "same", "score": 90, "reason": "manual", "conflicts": []}]},
            ]
        }
        with self.assertRaisesRegex(AssetReuseError, "重复使用确认"):
            _validate_reuse_selection(plan, {1: "same", 2: "same"}, set(), {"max_uses_per_asset": 1})
        payload = _validate_reuse_selection(plan, {1: "same", 2: "same"}, {2}, {"max_uses_per_asset": 1})
        self.assertTrue(payload["selections"][1]["duplicate_override"])

    def test_all_reused_produces_empty_supply_plan_and_merge_is_traceable(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        scene = {"shots": [_intent(1, "陶罐放在街角", "Roman jar")]}
        selection = {
            "selections": [
                {"shot_id": 1, "action": "reuse", "asset_id": "met-selected", "reuse_score": 100, "reuse_reason": "exact", "duplicate_override": False}
            ]
        }
        self.assertEqual(unmatched_scene_plan(scene, selection)["shots"], [])
        merged = merge_reused_candidates(scene, {"shots": []}, selection, snapshot, self.cache)
        candidate = merged["shots"][0]["candidates"][0]
        self.assertTrue(candidate["reused"])
        self.assertEqual(candidate["reused_from"]["parent_task_id"], "parent-task")

    def test_reused_ai_and_museum_assets_use_verified_cache_without_download(self) -> None:
        snapshot = create_reuse_source_snapshot(self.outputs, self.cache, "parent-task")
        entries = {item["asset_id"]: item for item in snapshot["assets"]}
        selections = []
        for shot_id, asset_id in ((1, "met-selected"), (2, "comfyui-one")):
            entry = entries[asset_id]
            candidate = dict(entry["candidate"])
            locator = entry["cache_locator"]
            local = (
                self.cache / "assets" / locator["value"]
                if locator["kind"] == "assets_sha256"
                else self.cache / "ai" / locator["value"]
            )
            candidate.update(
                {
                    "local_preview": str(local),
                    "reused": True,
                    "reuse_sha256": entry["sha256"],
                    "reused_from": {"parent_task_id": "parent-task", "asset_id": asset_id},
                    "reuse_score": 90,
                    "reuse_reason": "manual",
                }
            )
            selections.append(
                {
                    "shot_id": shot_id,
                    "intent_id": f"intent-{shot_id:03d}",
                    "narration": f"new-{shot_id}",
                    "candidate": candidate,
                }
            )
        run = Path(self.temp.name) / "derived-run"
        run.mkdir()
        with mock.patch("app.visual_supply.download_bytes", side_effect=AssertionError("unexpected network download")), mock.patch(
            "app.visual_supply._refetch_candidate", side_effect=lambda candidate, _: candidate
        ) as verifier:
            manifest = download_selected_assets(
                {"reviewed": True, "reviewed_at": "now", "selections": selections},
                {"min_long_edge": 1, "min_short_edge": 1},
                {},
                self.cache,
                run,
            )
        self.assertEqual(verifier.call_count, 2)
        self.assertTrue(all(item["reused"] for item in manifest["assets"]))
        self.assertEqual([item["narration"] for item in manifest["assets"]], ["new-1", "new-2"])


class ReuseTaskStateTests(unittest.TestCase):
    def test_derived_task_persists_dynamic_eighteen_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "task.json"
            task = TaskState.create(
                path,
                task_id="derived",
                run_id="run",
                project_id="project",
                input_hash="hash",
                stage_order=REUSE_TASK_STAGES,
            )
            self.assertEqual(len(task.stage_order), 18)
            task.begin("asset_reuse_match", "hash")
            task.succeed("asset_reuse_match", "hash")
            task.invalidate_after("asset_reuse_match")
            restored = TaskState.load(path)
            self.assertEqual(restored.stage_order, REUSE_TASK_STAGES)
            self.assertEqual(restored.data["stages"]["asset_reuse_review"]["status"], "pending")

    def test_render_cache_requires_identical_source_frames_motion_and_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.jpg"
            Image.new("RGB", (120, 180), "purple").save(source, format="JPEG")
            cache = root / "cache"
            canvas = {
                "width": 72,
                "height": 128,
                "fps": 30,
                "motion": {"working_scale": 1, "zoom_amount": 0.02, "easing": "cosine"},
            }
            shot = {
                "source": str(source),
                "kind": "image",
                "duration": 0.2,
                "frames": 6,
                "rendered_clip": "working/clips/shot-001.mp4",
                "fit": "crop",
                "focal_x": 0.5,
                "focal_y": 0.5,
                "motion": "zoom_in",
            }
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            render_shot(shot, first, canvas, {}, cache)
            original_run = pipeline._run

            def cached_only(args: object, **kwargs: object):
                if str(args[0]).lower().endswith("ffmpeg"):  # type: ignore[index]
                    raise AssertionError("cache miss")
                return original_run(args, **kwargs)  # type: ignore[arg-type]

            with mock.patch("app.pipeline._run", side_effect=cached_only):
                copied = render_shot(shot, second, canvas, {}, cache)
            self.assertTrue(copied.is_file())
            changed = {**shot, "frames": 7, "duration": 7 / 30}
            with mock.patch("app.pipeline._run", side_effect=cached_only):
                with self.assertRaisesRegex(AssertionError, "cache miss"):
                    render_shot(changed, second, canvas, {}, cache)


if __name__ == "__main__":
    unittest.main()
