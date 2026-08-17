from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.asset_review import ReviewError, prepare_and_deduplicate, validate_selection
from app.comfyui_client import ComfyUIError
from app.asset_sources import (
    AssetSourceError,
    normalize_met_object,
    normalize_http_url,
    normalize_openverse_result,
    normalize_rights,
    normalize_smithsonian_row,
    normalize_wikimedia_page,
    score_candidate,
    verify_image_bytes,
)
from app.deepseek_planner import DeepSeekPlannerError
from app.visual_semantics import review_asset_candidates
from app.visual_supply import (
    VisualSupplyError,
    add_ai_fallbacks,
    build_ai_only_candidates,
    build_sourced_storyboard,
    download_selected_assets,
    resolve_ai_image_limit,
    select_ai_candidate_targets,
    validate_asset_manifest,
    write_license_outputs,
)
from app.pipeline import write_ass
from app.task_state import TaskState
from app.transcript import Cue


FIXTURES = Path(__file__).parent / "fixtures" / "providers"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ProviderNormalizationTests(unittest.TestCase):
    def test_provider_url_with_spaces_is_percent_encoded(self):
        raw = "https://images.metmuseum.org/CRDImages/gr/original/67.265 8a-f.jpg"
        normalized = normalize_http_url(raw)
        self.assertEqual(
            normalized,
            "https://images.metmuseum.org/CRDImages/gr/original/67.265%208a-f.jpg",
        )
        self.assertNotIn(" ", normalized)
        with self.assertRaises(AssetSourceError):
            normalize_http_url("https://example.test/image.jpg\r\nX-Test: injected")

    def test_met_requires_public_domain_and_original(self):
        payload = fixture("met.json")
        self.assertEqual(normalize_met_object(payload["accepted"])["rights_code"], "pdm-1.0")
        self.assertIsNone(normalize_met_object(payload["rejected"]))

    def test_smithsonian_accepts_only_media_level_cc0(self):
        results = normalize_smithsonian_row(fixture("smithsonian.json"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rights_code"], "cc0-1.0")

    def test_wikimedia_requires_creator_source_and_strict_public_domain(self):
        payload = fixture("wikimedia.json")
        self.assertEqual(normalize_wikimedia_page(payload)["rights_code"], "pdm-1.0")
        payload["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = "CC BY-SA 4.0"
        payload["imageinfo"][0]["extmetadata"]["LicenseUrl"]["value"] = "https://creativecommons.org/licenses/by-sa/4.0/"
        self.assertIsNone(normalize_wikimedia_page(payload))

    def test_openverse_is_discovery_only(self):
        candidate = normalize_openverse_result(fixture("openverse.json"))
        self.assertIsNotNone(candidate)
        self.assertFalse(candidate["selectable"])
        self.assertTrue(candidate["requires_reverification"])

    def test_rejects_attribution_sharealike_nc_nd_and_unknown(self):
        for value in ("CC BY 4.0", "CC BY-SA", "CC BY-NC", "CC BY-ND", "All rights reserved", ""):
            self.assertIsNone(normalize_rights(value))

    def test_capture_year_is_not_a_hardcoded_rejection_rule(self):
        intent = {
            "era": "Ancient Rome",
            "search_terms_en": ["ancient Roman ammonia laundry"],
            "search_terms_zh": [],
            "objects": ["laundry vat"],
            "location": "Roman fullonica",
        }
        modern = {
            "title": "Andrena nubecula, ammonia, Maryland 2014",
            "creator": "Photographer",
            "institution": "Wikimedia Commons",
            "source_page": "https://example.test/insect",
            "rights_code": "pdm-1.0",
            "provider": "wikimedia",
            "width": 1800,
            "height": 1200,
            "selectable": True,
        }
        scored = score_candidate(modern, intent, "strict")
        self.assertTrue(scored["selectable"])
        self.assertGreater(scored["score"], 0)

        ancient = {
            **modern,
            "title": "Roman ceramic laundry vessel",
            "creator": "Unknown ancient maker",
            "source_page": "https://example.test/roman-vessel",
            "selectable": True,
        }
        self.assertTrue(score_candidate(ancient, intent, "strict")["selectable"])

    def test_full_decode_rejects_jpeg_missing_two_tail_bytes(self):
        import io

        stream = io.BytesIO()
        Image.new("RGB", (1024, 1536), (10, 20, 30)).save(stream, format="JPEG")
        with self.assertRaises(AssetSourceError):
            verify_image_bytes(
                stream.getvalue()[:-2],
                "image/jpeg",
                min_long_edge=1024,
                min_short_edge=640,
            )


class SemanticReviewTests(unittest.TestCase):
    def _scene_plan(self):
        return {
            "semantic_audit_status": "reviewed",
            "shots": [
                {
                    "shot_id": 1,
                    "intent_id": "intent-001",
                    "narration": "古罗马洗衣工使用石槽清洗衣物。",
                    "time_context": {
                        "label": "Ancient Rome",
                        "region": "Roman Empire",
                        "start_year": -100,
                        "end_year": 100,
                        "confidence": "approximate",
                        "period_terms_en": ["Ancient Rome"],
                        "period_terms_zh": ["古罗马"],
                    },
                    "location": "Roman fullonica",
                    "people": ["laundry worker"],
                    "objects": ["stone washing basin"],
                    "action": "washing cloth",
                    "must_include": ["Roman fullonica", "stone washing basin"],
                    "avoid": ["electric lighting", "plastic container"],
                }
            ],
        }

    def _candidate(self, asset_id: str, title: str, metadata: dict):
        return {
            "asset_id": asset_id,
            "provider": "wikimedia",
            "title": title,
            "creator": "Museum photographer",
            "institution": "Wikimedia Commons",
            "source_page": f"https://example.test/{asset_id}",
            "rights_code": "pdm-1.0",
            "width": 1800,
            "height": 2400,
            "selectable": True,
            "raw_metadata": metadata,
        }

    def _candidates(self):
        return {
            "shots": [
                {
                    "shot_id": 1,
                    "intent_id": "intent-001",
                    "recommended_asset_id": "insect-2014",
                    "candidates": [
                        self._candidate(
                            "insect-2014",
                            "Andrena nubecula, ammonia, Maryland 2014",
                            {"classification": "insect", "date": "2014"},
                        ),
                        self._candidate(
                            "roman-object-2014",
                            "Roman fullonica stone washing basin photographed 2014",
                            {"culture": "Roman", "period": "Imperial", "date": "2014"},
                        ),
                    ],
                }
            ]
        }

    def _config(self):
        return {
            "semantic_review": {
                "enabled": True,
                "provider": "deepseek",
                "shots_per_batch": 4,
                "reject_below": 40,
                "recommendation_threshold": 75,
                "failure_policy": "manual_review",
            }
        }

    def test_semantic_review_rejects_keyword_false_match_not_capture_year(self):
        response = {
            "judgments": [
                {
                    "shot_id": 1,
                    "asset_id": "insect-2014",
                    "relevance_score": 8,
                    "period_score": 5,
                    "subject_score": 0,
                    "visual_fit_score": 20,
                    "verdict": "reject",
                    "conflicts": ["昆虫照片与古罗马洗衣工艺无关"],
                    "reason": "仅因 ammonia 一词误匹配",
                },
                {
                    "shot_id": 1,
                    "asset_id": "roman-object-2014",
                    "relevance_score": 92,
                    "period_score": 95,
                    "subject_score": 90,
                    "visual_fit_score": 80,
                    "verdict": "eligible",
                    "conflicts": [],
                    "reason": "拍摄年份不影响其所描绘的古罗马文物",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_semantics.request_json_object",
            return_value=(response, {"id": "semantic-test"}),
        ) as request:
            reviewed, report, _ = review_asset_candidates(
                self._scene_plan(),
                self._candidates(),
                {"model": "test-model"},
                self._config(),
                {"DEEPSEEK_API_KEY": "test"},
                Path(temporary),
            )
            second, _, _ = review_asset_candidates(
                self._scene_plan(),
                self._candidates(),
                {"model": "test-model"},
                self._config(),
                {"DEEPSEEK_API_KEY": "test"},
                Path(temporary),
            )
        rows = {item["asset_id"]: item for item in reviewed["shots"][0]["candidates"]}
        self.assertEqual(rows["insect-2014"]["semantic_status"], "rejected")
        self.assertEqual(rows["roman-object-2014"]["semantic_status"], "reviewed")
        self.assertEqual(reviewed["shots"][0]["recommended_asset_id"], "roman-object-2014")
        self.assertEqual(second["shots"][0]["recommended_asset_id"], "roman-object-2014")
        self.assertEqual(report["status"], "reviewed")
        self.assertEqual(request.call_count, 1)

    def test_semantic_service_unavailable_forces_manual_choice(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_semantics.request_json_object",
            side_effect=DeepSeekPlannerError("service unavailable"),
        ):
            reviewed, report, _ = review_asset_candidates(
                self._scene_plan(),
                self._candidates(),
                {"model": "test-model"},
                self._config(),
                {"DEEPSEEK_API_KEY": "test"},
                Path(temporary),
            )
        self.assertEqual(report["status"], "unavailable")
        self.assertIsNone(reviewed["shots"][0]["recommended_asset_id"])
        self.assertTrue(
            all(
                item["semantic_status"] == "unavailable"
                for item in reviewed["shots"][0]["candidates"]
            )
        )

    def test_high_quality_venus_is_still_rejected_for_laundry_shot(self):
        candidates = self._candidates()
        candidates["shots"][0]["candidates"] = [
            self._candidate(
                "venus-cupid",
                "Venus and Cupid",
                {"culture": "European", "classification": "painting", "medium": "oil"},
            )
        ]
        response = {
            "judgments": [
                {
                    "shot_id": 1,
                    "asset_id": "venus-cupid",
                    "relevance_score": 15,
                    "period_score": 25,
                    "subject_score": 0,
                    "visual_fit_score": 90,
                    "verdict": "reject",
                    "conflicts": ["神话绘画不包含洗衣作坊或石制洗衣槽"],
                    "reason": "机构与画质不能替代主题相关性",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_semantics.request_json_object",
            return_value=(response, {"id": "semantic-test"}),
        ):
            reviewed, _, _ = review_asset_candidates(
                self._scene_plan(),
                candidates,
                {"model": "test-model"},
                self._config(),
                {"DEEPSEEK_API_KEY": "test"},
                Path(temporary),
            )
        candidate = reviewed["shots"][0]["candidates"][0]
        self.assertEqual(candidate["semantic_status"], "rejected")
        self.assertIsNone(reviewed["shots"][0]["recommended_asset_id"])


class ReviewAndFallbackTests(unittest.TestCase):
    def _candidates(self):
        def item(asset_id: str):
            return {
                "asset_id": asset_id,
                "selectable": True,
                "provider": "met",
                "title": asset_id,
                "creator": "x",
                "source_page": "https://example.test/item",
            }

        return {
            "shots": [
                {"shot_id": 1, "intent_id": "i1", "candidates": [item("a")]},
                {"shot_id": 2, "intent_id": "i2", "candidates": [item("a"), item("b")]},
            ]
        }

    def test_review_rejects_unknown_and_adjacent_reuse(self):
        with self.assertRaises(ReviewError):
            validate_selection(self._candidates(), {1: "a", 2: "missing"})
        with self.assertRaises(ReviewError):
            validate_selection(self._candidates(), {1: "a", 2: "a"})
        result = validate_selection(self._candidates(), {1: "a", 2: "b"})
        self.assertTrue(result["reviewed"])

    def test_semantic_rejection_requires_explicit_override_but_rights_never_can(self):
        candidates = self._candidates()
        low = candidates["shots"][0]["candidates"][0]
        low.update(
            {
                "semantic_status": "rejected",
                "semantic_requires_override": True,
                "semantic_score": 12,
                "semantic_review": {
                    "reason": "主题不相关",
                    "conflicts": ["错误物件"],
                },
            }
        )
        with self.assertRaisesRegex(ReviewError, "风险确认"):
            validate_selection(candidates, {1: "a", 2: "b"})
        accepted = validate_selection(
            candidates, {1: "a", 2: "b"}, semantic_overrides={1}
        )
        self.assertTrue(accepted["selections"][0]["semantic_override"])
        low["selectable"] = False
        with self.assertRaisesRegex(ReviewError, "授权核验"):
            validate_selection(
                candidates, {1: "a", 2: "b"}, semantic_overrides={1}
            )

    def test_thumbnail_perceptual_duplicates_are_collapsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "same.jpg"
            Image.new("RGB", (800, 1200), (90, 60, 30)).save(image)
            candidates = {
                "shots": [{
                    "shot_id": 1,
                    "recommended_asset_id": "a",
                    "candidates": [
                        {"asset_id": "a", "local_preview": str(image), "selectable": True, "score": 90},
                        {"asset_id": "b", "local_preview": str(image), "selectable": True, "score": 80},
                    ],
                }]
            }
            prepare_and_deduplicate(root, candidates, 70)
            self.assertTrue(candidates["shots"][0]["candidates"][0]["selectable"])
            self.assertFalse(candidates["shots"][0]["candidates"][1]["selectable"])

    def test_ai_limit_fails_before_any_api_call(self):
        candidates = {
            "at_least_one_source_succeeded": True,
            "shots": [{"shot_id": i, "recommended_asset_id": None} for i in range(1, 6)],
        }
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(VisualSupplyError) as caught:
            add_ai_fallbacks(
                candidates,
                {"shots": []},
                {"enabled": True, "max_images_per_run": 4},
                "",
                Path(temporary),
                Path(temporary) / "provenance",
            )
        self.assertIn("任何付费调用发生前", str(caught.exception))

    def test_local_comfyui_limit_covers_all_shots_and_review_headroom(self):
        config = {
            "provider": "comfyui_local",
            "max_images_per_run": "all_shots",
            "regeneration_headroom": 4,
        }
        self.assertEqual(resolve_ai_image_limit(config, "comfyui_local", 16), 20)
        with self.assertRaises(VisualSupplyError):
            resolve_ai_image_limit(config, "openai", 16)
        four_each = {
            "provider": "comfyui_local",
            "candidates_per_shot": 4,
            "max_images_per_run": "all_candidates",
            "regeneration_headroom": 4,
        }
        self.assertEqual(resolve_ai_image_limit(four_each, "comfyui_local", 25), 104)

    def test_ai_only_candidate_envelope_never_calls_museum_search(self):
        scene_plan = {
            "shots": [
                {
                    "shot_id": 1,
                    "intent_id": "intent-1",
                    "narration": "历史场景",
                    "time_context": {"label": "Ancient"},
                    "must_include": ["pottery"],
                    "avoid": ["plastic"],
                }
            ]
        }
        with patch("app.visual_supply.search_assets") as search:
            candidates = build_ai_only_candidates(
                scene_plan, {"providers": ["met", "wikimedia"]}
            )
        search.assert_not_called()
        self.assertTrue(candidates["search_skipped"])
        self.assertEqual(candidates["visual_strategy"], "ai_only")
        self.assertEqual(candidates["shots"][0]["candidates"], [])
        self.assertTrue(
            all(
                status["status"] == "skipped"
                for status in candidates["provider_status"].values()
            )
        )

    def test_ai_only_generates_four_distinct_candidates_and_resumes_partial_cache(self):
        stream = io.BytesIO()
        Image.new("RGB", (768, 1344), (30, 50, 70)).save(stream, format="JPEG")
        image_bytes = stream.getvalue()
        scene_plan = {
            "shots": [
                {
                    "shot_id": 1,
                    "intent_id": "intent-1",
                    "ai_prompt": "Historical workshop.",
                    "must_include": ["stone basin"],
                    "avoid": ["plastic"],
                }
            ]
        }
        config = {
            "enabled": True,
            "provider": "comfyui_local",
            "candidate_policy": "all_shots",
            "candidates_per_shot": 4,
            "max_images_per_run": "all_candidates",
            "regeneration_headroom": 0,
            "model_license_url": "https://example.test/license",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidates.json"
            candidates = build_ai_only_candidates(scene_plan, {})
            calls: list[int] = []

            def fail_on_third(prompt, config, project_dir, *, seed):
                calls.append(seed)
                if len(calls) == 3:
                    raise ComfyUIError("simulated interruption")
                return image_bytes, {"request_id": f"local-{len(calls)}", "seed": seed}

            with patch(
                "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
            ), patch(
                "app.visual_supply.generate_comfyui_image",
                side_effect=fail_on_third,
            ):
                with self.assertRaises(VisualSupplyError):
                    add_ai_fallbacks(
                        candidates,
                        scene_plan,
                        config,
                        {},
                        root / "cache",
                        root / "provenance",
                        root,
                        checkpoint,
                    )
            partial = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(len(partial["shots"][0]["candidates"]), 2)
            resumed_calls: list[int] = []

            def finish(prompt, config, project_dir, *, seed):
                resumed_calls.append(seed)
                return image_bytes, {"request_id": f"resume-{len(resumed_calls)}", "seed": seed}

            with patch(
                "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
            ), patch(
                "app.visual_supply.generate_comfyui_image", side_effect=finish
            ):
                completed, _ = add_ai_fallbacks(
                    partial,
                    scene_plan,
                    config,
                    {},
                    root / "cache",
                    root / "provenance",
                    root,
                    checkpoint,
                )
            rows = completed["shots"][0]["candidates"]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(resumed_calls), 2)
            self.assertEqual(len({row["asset_id"] for row in rows}), 4)
            self.assertEqual(len({row["generation"]["seed"] for row in rows}), 4)
            self.assertIsNone(completed["shots"][0]["recommended_asset_id"])

    def test_ai_candidate_policy_can_target_every_shot_without_replacing_museum_choice(self):
        candidates = {
            "shots": [
                {"shot_id": 1, "recommended_asset_id": "met-a"},
                {"shot_id": 2, "recommended_asset_id": None},
            ]
        }
        self.assertEqual(
            [item["shot_id"] for item in select_ai_candidate_targets(candidates, "gaps")],
            [2],
        )
        self.assertEqual(
            [item["shot_id"] for item in select_ai_candidate_targets(candidates, "all_shots")],
            [1, 2],
        )

    def test_all_shots_policy_adds_ai_option_and_preserves_museum_recommendation(self):
        stream = io.BytesIO()
        Image.new("RGB", (768, 1344), (20, 40, 60)).save(stream, format="JPEG")
        image_bytes = stream.getvalue()
        candidates = {
            "at_least_one_source_succeeded": True,
            "shots": [
                {
                    "shot_id": 1,
                    "recommended_asset_id": "met-a",
                    "museum_source_succeeded": True,
                    "candidates": [{"asset_id": "met-a", "provider": "met"}],
                },
                {
                    "shot_id": 2,
                    "recommended_asset_id": None,
                    "museum_source_succeeded": True,
                    "candidates": [],
                },
            ],
        }
        scene_plan = {
            "shots": [
                {"shot_id": 1, "ai_prompt": "Ancient Roman ceramic vessel."},
                {"shot_id": 2, "ai_prompt": "Ancient Roman laundry workshop."},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
        ), patch(
            "app.visual_supply.generate_comfyui_image",
            return_value=(image_bytes, {"request_id": "local-test", "model": "test"}),
        ):
            updated, _ = add_ai_fallbacks(
                candidates,
                scene_plan,
                {
                    "enabled": True,
                    "provider": "comfyui_local",
                    "candidate_policy": "all_shots",
                    "max_images_per_run": "all_shots",
                    "regeneration_headroom": 0,
                    "model_license_url": "https://example.test/license",
                },
                {},
                Path(temporary) / "cache",
                Path(temporary) / "provenance",
                Path(temporary),
            )
        self.assertEqual(updated["shots"][0]["recommended_asset_id"], "met-a")
        self.assertEqual(updated["shots"][0]["candidates"][0]["provider"], "comfyui_local")
        self.assertTrue(updated["shots"][1]["recommended_asset_id"].startswith("comfyui-"))
        self.assertEqual(updated["ai_generated_count"], 2)

    def test_semantic_outage_keeps_ai_candidate_unchecked_for_manual_review(self):
        stream = io.BytesIO()
        Image.new("RGB", (768, 1344), (20, 40, 60)).save(stream, format="JPEG")
        candidates = {
            "at_least_one_source_succeeded": True,
            "semantic_review_status": "unavailable",
            "shots": [
                {
                    "shot_id": 1,
                    "recommended_asset_id": None,
                    "semantic_review_status": "unavailable",
                    "museum_source_succeeded": True,
                    "candidates": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
        ), patch(
            "app.visual_supply.generate_comfyui_image",
            return_value=(stream.getvalue(), {"request_id": "local-test", "model": "test"}),
        ):
            updated, _ = add_ai_fallbacks(
                candidates,
                {"shots": [{"shot_id": 1, "ai_prompt": "Historical scene.", "must_include": [], "avoid": []}]},
                {
                    "enabled": True,
                    "provider": "comfyui_local",
                    "candidate_policy": "all_shots",
                    "max_images_per_run": "all_shots",
                    "regeneration_headroom": 0,
                    "model_license_url": "https://example.test/license",
                },
                {},
                Path(temporary) / "cache",
                Path(temporary) / "provenance",
                Path(temporary),
            )
        self.assertIsNone(updated["shots"][0]["recommended_asset_id"])
        self.assertEqual(
            updated["shots"][0]["candidates"][0]["semantic_status"],
            "ai_unreviewed",
        )

    def test_search_outage_does_not_trigger_ai(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(VisualSupplyError):
            add_ai_fallbacks(
                {"at_least_one_source_succeeded": False, "shots": [{"shot_id": 1}]},
                {"shots": []},
                {"enabled": True, "max_images_per_run": 4},
                "",
                Path(temporary),
                Path(temporary) / "provenance",
            )

    def test_local_all_shots_can_continue_through_partial_museum_outage(self):
        stream = io.BytesIO()
        Image.new("RGB", (768, 1344), (50, 70, 90)).save(stream, format="JPEG")
        candidates = {
            "at_least_one_source_succeeded": True,
            "shots": [
                {
                    "shot_id": 16,
                    "recommended_asset_id": None,
                    "museum_source_succeeded": False,
                    "successful_providers": [],
                    "candidates": [],
                }
            ],
        }
        scene_plan = {
            "shots": [
                {
                    "shot_id": 16,
                    "ai_prompt": "A historically accurate workshop interior.",
                    "must_include": ["stone basin"],
                    "avoid": ["modern machinery"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
        ), patch(
            "app.visual_supply.generate_comfyui_image",
            return_value=(stream.getvalue(), {"request_id": "local-test", "model": "test"}),
        ):
            updated, _ = add_ai_fallbacks(
                candidates,
                scene_plan,
                {
                    "enabled": True,
                    "provider": "comfyui_local",
                    "candidate_policy": "all_shots",
                    "candidates_per_shot": 1,
                    "max_images_per_run": "all_candidates",
                    "regeneration_headroom": 0,
                    "model_license_url": "https://example.test/license",
                },
                {},
                Path(temporary) / "cache",
                Path(temporary) / "provenance",
                Path(temporary),
            )
        self.assertEqual(updated["ai_generated_count"], 1)
        self.assertTrue(updated["shots"][0]["recommended_asset_id"].startswith("comfyui-"))
        self.assertEqual(
            updated["museum_outage_ai_override"]["shot_ids"],
            [16],
        )

    def test_local_all_shots_can_continue_when_every_museum_source_is_down(self):
        stream = io.BytesIO()
        Image.new("RGB", (768, 1344), (50, 70, 90)).save(stream, format="JPEG")
        candidates = {
            "at_least_one_source_succeeded": False,
            "shots": [
                {
                    "shot_id": 1,
                    "recommended_asset_id": None,
                    "museum_source_succeeded": False,
                    "successful_providers": [],
                    "candidates": [],
                }
            ],
        }
        scene_plan = {
            "shots": [
                {"shot_id": 1, "ai_prompt": "Historical scene.", "must_include": [], "avoid": []}
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.visual_supply.workflow_fingerprint", return_value="workflow-sha"
        ), patch(
            "app.visual_supply.generate_comfyui_image",
            return_value=(stream.getvalue(), {"request_id": "local-test", "model": "test"}),
        ):
            updated, _ = add_ai_fallbacks(
                candidates,
                scene_plan,
                {
                    "enabled": True,
                    "provider": "comfyui_local",
                    "candidate_policy": "all_shots",
                    "candidates_per_shot": 1,
                    "max_images_per_run": "all_candidates",
                    "regeneration_headroom": 0,
                    "model_license_url": "https://example.test/license",
                },
                {},
                Path(temporary) / "cache",
                Path(temporary) / "provenance",
                Path(temporary),
            )
        self.assertEqual(updated["ai_generated_count"], 1)
        self.assertEqual(updated["museum_outage_ai_override"]["shot_ids"], [1])

    def test_museum_outage_still_blocks_local_gap_and_external_all_shots(self):
        candidates = {
            "at_least_one_source_succeeded": True,
            "shots": [
                {
                    "shot_id": 1,
                    "recommended_asset_id": None,
                    "museum_source_succeeded": False,
                    "candidates": [],
                }
            ],
        }
        scene_plan = {
            "shots": [
                {"shot_id": 1, "ai_prompt": "Historical scene.", "must_include": [], "avoid": []}
            ]
        }
        for config in (
            {
                "enabled": True,
                "provider": "comfyui_local",
                "candidate_policy": "gaps",
                "max_images_per_run": 4,
            },
            {
                "enabled": True,
                "provider": "openai",
                "candidate_policy": "all_shots",
                "max_images_per_run": 4,
            },
        ):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(VisualSupplyError):
                    add_ai_fallbacks(
                        candidates,
                        scene_plan,
                        config,
                        {},
                        Path(temporary) / "cache",
                        Path(temporary) / "provenance",
                    )


class LedgerAndStoryboardTests(unittest.TestCase):
    def test_manifest_reuse_rejects_truncated_image(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.BytesIO()
            Image.new("RGB", (1024, 1536), (20, 30, 40)).save(stream, format="JPEG")
            data = stream.getvalue()[:-2]
            image = root / "shot-001-broken.jpg"
            image.write_bytes(data)
            manifest = {
                "assets": [{
                    "local_path": str(image),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "mime": "image/jpeg",
                    "width": 1024,
                    "height": 1536,
                }]
            }
            with self.assertRaises(VisualSupplyError):
                validate_asset_manifest(
                    manifest, {"min_long_edge": 1024, "min_short_edge": 640}
                )

    def test_asset_download_resume_reuses_verified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            cache = Path(temporary) / "cache"
            assets = run / "assets"
            provenance = run / "provenance"
            working = run / "working"
            assets.mkdir(parents=True)
            provenance.mkdir()
            working.mkdir()
            image = assets / "shot-001-placeholder.jpg"
            Image.new("RGB", (1024, 1536), (20, 30, 40)).save(image)
            data = image.read_bytes()
            import hashlib

            digest = hashlib.sha256(data).hexdigest()
            final_image = assets / f"shot-001-{digest[:12]}.jpg"
            image.replace(final_image)
            candidate = {
                "asset_id": "comfyui-existing",
                "provider": "comfyui_local",
                "source_id": "prompt",
                "title": "Existing AI image",
                "creator": "Local",
                "institution": "Local ComfyUI",
                "source_page": "http://127.0.0.1:8000/",
                "download_url": "",
                "rights_code": "provider_terms",
                "rights_url": "https://example.test/model-license",
                "ai_generated": True,
                "score": 70,
                "generation": {"model": "test"},
            }
            (provenance / "comfyui-existing.json").write_text(
                json.dumps(
                    {
                        "review_candidate": candidate,
                        "download_verification": {
                            "verified_at": "now",
                            "mime": "image/jpeg",
                            "width": 1024,
                            "height": 1536,
                            "sha256": digest,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = download_selected_assets(
                {
                    "reviewed": True,
                    "reviewed_at": "now",
                    "selections": [
                        {
                            "shot_id": 1,
                            "intent_id": "intent-001",
                            "narration": "test",
                            "candidate": candidate,
                        }
                    ],
                },
                {"min_long_edge": 1024, "min_short_edge": 640},
                {},
                cache,
                run,
            )
            self.assertEqual(len(manifest["assets"]), 1)
            self.assertEqual(manifest["assets"][0]["sha256"], digest)

    def test_ledger_and_storyboard_v2_are_traceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "asset.jpg"
            Image.new("RGB", (1024, 1536), (20, 30, 40)).save(image)
            manifest = {
                "human_reviewed": True,
                "assets": [{
                    "shot_id": 1, "intent_id": "intent-001", "narration": "一只陶罐",
                    "asset_id": "met-a", "provider": "met", "title": "Jar",
                    "creator": "Unknown", "institution": "The Met", "collection_id": "1",
                    "source_page": "https://example.test/1", "download_url": "https://example.test/a.jpg",
                    "rights_code": "pdm-1.0", "rights_url": "https://creativecommons.org/publicdomain/mark/1.0/",
                    "retrieved_at": "now", "reviewed_at": "now", "sha256": "abc",
                    "width": 1024, "height": 1536, "local_path": str(image),
                    "provenance_ref": "provenance/met-a.json", "ai_generated": False, "score": 88,
                    "semantic_status": "reviewed", "semantic_score": 92,
                    "semantic_reason": "时代与物件均相关", "semantic_override": False,
                }],
            }
            (root / "provenance").mkdir()
            (root / "provenance" / "met-a.json").write_text("{}", encoding="utf-8")
            _, _, _, audit = write_license_outputs(manifest, root)
            self.assertTrue(audit["asset_rights_ready"])
            self.assertEqual(audit["semantic_status_counts"], {"reviewed": 1})
            self.assertEqual(audit["semantic_override_count"], 0)
            storyboard = build_sourced_storyboard(
                "test", 2.0, {"width": 1080, "height": 1920, "fps": 30},
                [{"start": 0.0, "end": 2.0, "duration": 2.0, "frames": 60, "caption_text": "一只陶罐"}],
                manifest,
            )
            self.assertEqual(storyboard["schema_version"], 2)
            self.assertEqual(storyboard["shots"][0]["provenance_ref"], "provenance/met-a.json")
            self.assertEqual(storyboard["shots"][0]["rights_code"], "pdm-1.0")
            self.assertEqual(storyboard["shots"][0]["semantic_score"], 92)
            self.assertFalse(storyboard["shots"][0]["semantic_override"])

    def test_ai_disclosure_is_added_without_extending_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captions.ass"
            write_ass(
                path,
                [Cue(0.0, 1.0, "测试字幕")],
                {"width": 1080, "height": 1920},
                {
                    "font_name": "Microsoft YaHei", "font_size": 52,
                    "base_color": "#FFFFFF", "highlight_color": "#FFD54A",
                    "outline": 4, "shadow": 1, "margin_bottom": 340,
                    "style_preset": "history_keyword",
                },
                {"cues": []},
                disclosure={
                    "required": True, "text": "部分画面为 AI 历史重构",
                    "seconds": 2.0, "end": 10.0,
                },
            )
            content = path.read_text(encoding="utf-8-sig")
            self.assertIn("Dialogue: 3,0:00:08.00,0:00:10.00,Disclosure", content)
            self.assertIn("部分画面为 AI 历史重构", content)

    def test_waiting_for_review_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            task = TaskState.create(
                path,
                task_id="task", run_id="run", project_id="project", input_hash="hash",
            )
            task.begin("asset_review", "hash")
            task.wait_for_review("asset_review", "hash", [])
            reloaded = TaskState.load(path)
            self.assertEqual(reloaded.data["status"], "waiting_for_review")
            self.assertEqual(
                reloaded.data["stages"]["asset_review"]["status"], "waiting_for_review"
            )

    def test_repairing_upstream_asset_invalidates_downstream_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            task = TaskState.create(
                path,
                task_id="task",
                run_id="run",
                project_id="project",
                input_hash="hash",
            )
            task.succeed("asset_download", "hash")
            task.succeed("license_audit", "hash")
            task.succeed("storyboard", "hash")
            task.invalidate_after("asset_download")
            self.assertEqual(task.data["stages"]["asset_download"]["status"], "succeeded")
            self.assertEqual(task.data["stages"]["license_audit"]["status"], "pending")
            self.assertEqual(task.data["stages"]["storyboard"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
