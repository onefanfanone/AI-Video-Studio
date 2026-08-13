from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from app.comfyui_client import (
    ComfyUIError,
    generate_image as generate_comfyui_image,
    validate_server_url,
)
from app.deepseek_planner import (
    _validate_plan,
    audit_scene_plan,
    create_scene_plan as create_deepseek_plan,
)
from unittest.mock import patch


def _start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _intent(shot_id: int) -> dict:
    return {
        "intent_id": f"model-{shot_id}",
        "shot_id": shot_id,
        "era": "Ancient Rome",
        "time_context": {
            "label": "Ancient Rome",
            "region": "Roman Empire",
            "start_year": -100,
            "end_year": 100,
            "confidence": "approximate",
            "period_terms_en": ["Ancient Rome", "Roman Imperial"],
            "period_terms_zh": ["古罗马", "罗马帝国"],
        },
        "location": "Roman street",
        "people": ["pedestrian"],
        "objects": ["ceramic jar"],
        "action": "walking past a jar",
        "mood": "curious",
        "search_terms_zh": ["古罗马 陶罐"],
        "search_terms_en": ["ancient Roman ceramic vessel museum"],
        "must_include": ["ceramic vessel"],
        "avoid": ["modern objects"],
        "ai_prompt": "Vertical Ancient Roman street with a ceramic vessel.",
    }


def _deepseek_raw(shots: list[dict], response_id: str = "chat-test") -> dict:
    return {
        "id": response_id,
        "model": "test-model",
        "created": 1,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"shots": shots}, ensure_ascii=False),
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


class _DeepSeekHandler(BaseHTTPRequestHandler):
    request_payload = None
    authorization = None

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        cls = type(self)
        cls.authorization = self.headers.get("Authorization")
        length = int(self.headers.get("Content-Length", "0"))
        cls.request_payload = json.loads(self.rfile.read(length).decode("utf-8"))
        content = json.dumps({"shots": [_intent(1), _intent(2)]}, ensure_ascii=False)
        body = json.dumps(
            {
                "id": "chat-test",
                "model": "test-model",
                "created": 1,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ComfyHandler(BaseHTTPRequestHandler):
    prompt_payload = None

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        cls = type(self)
        length = int(self.headers.get("Content-Length", "0"))
        cls.prompt_payload = json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps({"prompt_id": "prompt-test"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/object_info":
            body = json.dumps(
                {
                    "CLIPTextEncode": {},
                    "EmptySD3LatentImage": {},
                    "KSampler": {},
                    "SaveImage": {},
                }
            ).encode("utf-8")
            content_type = "application/json"
        elif path == "/history/prompt-test":
            body = json.dumps(
                {
                    "prompt-test": {
                        "outputs": {
                            "4": {
                                "images": [
                                    {"filename": "test.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                        "status": {"completed": True, "status_str": "success"},
                    }
                }
            ).encode("utf-8")
            content_type = "application/json"
        elif path == "/view":
            stream = io.BytesIO()
            Image.new("RGB", (768, 1344), (20, 40, 60)).save(stream, format="PNG")
            body = stream.getvalue()
            content_type = "image/png"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DeepSeekPlannerTests(unittest.TestCase):
    def test_dynamic_validation_accepts_explicit_english_exclusions(self):
        intent = _intent(1)
        intent["avoid"] = ["modern objects", "text", "watermark"]
        intent["ai_prompt"] = (
            "Ancient Roman street with period clothing. No modern objects, "
            "no laboratory equipment, no text, and no watermark."
        )
        validated = _validate_plan({"shots": [intent]}, 1, "strict")
        self.assertEqual(validated[0]["time_context"]["start_year"], -100)

    def test_dynamic_validation_accepts_chinese_exclusion(self):
        intent = _intent(1)
        intent["avoid"] = ["不要智能手机", "文字", "水印"]
        intent["ai_prompt"] = "Ancient Rome street, 不要出现智能手机，避免文字和水印。"
        validated = _validate_plan({"shots": [intent]}, 1, "strict")
        self.assertEqual(validated[0]["shot_id"], 1)

    def test_dynamic_validation_rejects_affirmative_forbidden_element(self):
        intent = _intent(1)
        intent["avoid"] = ["smartphone"]
        intent["objects"] = ["ceramic jar", "smartphone"]
        intent["ai_prompt"] = "Ancient Rome street with a ceramic jar and smartphone."
        with self.assertRaisesRegex(ValueError, "禁用元素"):
            _validate_plan({"shots": [intent]}, 1, "strict")

    def test_phrase_level_avoid_does_not_reject_related_historical_subject(self):
        cases = [
            (
                "washing basin",
                "modern washing machines",
                "worker washing cloth in a stone washing basin",
            ),
            (
                "Roman public restroom",
                "public restroom signs",
                "Roman public restroom entrance without signage",
            ),
            (
                "Roman coin",
                "paper money",
                "Roman money represented by a metal coin",
            ),
            (
                "oil lamp",
                "light bulbs",
                "warm light from an oil lamp",
            ),
            (
                "imperial administrative building",
                "government buildings with glass",
                "stone government building facade",
            ),
        ]
        for subject, forbidden, description in cases:
            with self.subTest(forbidden=forbidden):
                intent = _intent(1)
                intent["objects"] = [subject]
                intent["must_include"] = [subject]
                intent["avoid"] = [forbidden]
                intent["search_terms_en"] = [f"Ancient Rome {subject} museum"]
                intent["ai_prompt"] = f"Vertical Ancient Rome scene, {description}."
                validated = _validate_plan({"shots": [intent]}, 1, "strict")
                self.assertEqual(validated[0]["must_include"], [subject])

    def test_dynamic_periods_need_no_python_allowlist(self):
        periods = [
            ("Edo period", "Tokugawa Japan", 1603, 1868),
            ("Aztec Empire", "Mesoamerica", 1428, 1521),
            ("1920s Shanghai", "Shanghai", 1920, 1929),
            ("World War II", "Europe", 1939, 1945),
            ("Early computing", "United States", 1940, 1955),
        ]
        for label, region, start, end in periods:
            with self.subTest(label=label):
                intent = _intent(1)
                intent["era"] = label
                intent["time_context"] = {
                    "label": label,
                    "region": region,
                    "start_year": start,
                    "end_year": end,
                    "confidence": "approximate",
                    "period_terms_en": [label],
                    "period_terms_zh": [],
                }
                intent["location"] = region
                intent["objects"] = ["document"]
                intent["must_include"] = ["document"]
                intent["avoid"] = ["unrelated object"]
                intent["search_terms_en"] = [f"{label} document museum"]
                intent["search_terms_zh"] = []
                intent["ai_prompt"] = f"Vertical {label} scene with a document."
                validated = _validate_plan({"shots": [intent]}, 1, "strict")
                self.assertEqual(validated[0]["time_context"]["label"], label)

    def test_json_plan_is_validated_and_normalized(self):
        server, thread = _start_server(_DeepSeekHandler)
        try:
            plan, safe = create_deepseek_plan(
                "测试文案",
                [
                    {"start": 0.0, "end": 1.5, "caption_text": "第一镜"},
                    {"start": 1.5, "end": 3.0, "caption_text": "第二镜"},
                ],
                {
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                    "model": "test-model",
                    "thinking": "disabled",
                },
                "test-key",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(_DeepSeekHandler.authorization, "Bearer test-key")
        self.assertEqual(_DeepSeekHandler.request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(_DeepSeekHandler.request_payload["thinking"], {"type": "disabled"})
        self.assertEqual(plan["provider"], "deepseek")
        self.assertEqual(plan["shots"][0]["intent_id"], "intent-001")
        self.assertEqual(plan["shots"][1]["narration"], "第二镜")
        self.assertEqual(len(plan["local_normalizations"]), 2)
        self.assertEqual(safe["finish_reason"], "stop")

    def test_invalid_shot_is_repaired_without_rewriting_valid_shots(self):
        broken = _intent(1)
        broken["avoid"] = ["smartphone"]
        broken["objects"] = ["ceramic jar", "smartphone"]
        broken["ai_prompt"] = "Ancient Rome street with a ceramic jar and smartphone."
        valid_second = _intent(2)
        valid_second["mood"] = "preserve-this-valid-shot"
        repaired = _intent(1)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "app.deepseek_planner._post_json",
            side_effect=[
                _deepseek_raw([broken, valid_second], "initial"),
                _deepseek_raw([], "bad-partial-repair"),
                _deepseek_raw([repaired], "good-targeted-repair"),
            ],
        ) as post:
            diagnostics = Path(temporary) / "attempts.json"
            plan, safe = create_deepseek_plan(
                "测试文案",
                [
                    {"start": 0.0, "end": 1.5, "caption_text": "第一镜"},
                    {"start": 1.5, "end": 3.0, "caption_text": "第二镜"},
                ],
                {"base_url": "https://example.test", "model": "test-model"},
                "test-key",
                diagnostics_path=diagnostics,
            )
            recorded = json.loads(diagnostics.read_text(encoding="utf-8"))
        self.assertEqual(post.call_count, 3)
        self.assertEqual(len(plan["shots"]), 2)
        self.assertEqual(plan["shots"][1]["mood"], "preserve-this-valid-shot")
        self.assertNotIn("smartphone", plan["shots"][0]["objects"])
        self.assertEqual(safe["id"], "good-targeted-repair")
        self.assertEqual(
            [item["phase"] for item in recorded["attempts"]],
            ["initial_plan", "targeted_repair", "targeted_repair"],
        )
        repair_request = json.loads(post.call_args_list[1].args[1]["messages"][1]["content"])
        self.assertEqual(repair_request["requested_shot_ids"], [1])
        self.assertEqual(len(repair_request["invalid_shots"]), 1)

    def test_wrong_full_shot_count_retries_without_truncated_assistant_history(self):
        calls = []

        def respond(_url, payload, _key, *, timeout):
            calls.append(payload)
            if len(calls) == 1:
                return _deepseek_raw([_intent(1)], "short-plan")
            return _deepseek_raw([_intent(1), _intent(2)], "complete-plan")

        with patch("app.deepseek_planner._post_json", side_effect=respond):
            plan, _ = create_deepseek_plan(
                "测试文案",
                [
                    {"start": 0.0, "end": 1.5, "caption_text": "第一镜"},
                    {"start": 1.5, "end": 3.0, "caption_text": "第二镜"},
                ],
                {"base_url": "https://example.test", "model": "test-model"},
                "test-key",
            )
        self.assertEqual(len(plan["shots"]), 2)
        self.assertEqual(len(calls[1]["messages"]), 2)
        self.assertTrue(all(item["role"] != "assistant" for item in calls[1]["messages"]))
        retry_request = json.loads(calls[1]["messages"][1]["content"])
        self.assertIn("本次返回 1", retry_request["previous_validation_error"])

    def test_plan_audit_repairs_dynamic_conflict_then_passes(self):
        plan = {"schema_version": 2, "shots": [_intent(1)]}
        repaired = _intent(1)
        repaired["objects"] = ["stone washing basin"]
        repaired["must_include"] = ["stone washing basin"]
        repaired["search_terms_en"] = ["Ancient Rome stone washing basin museum"]
        repaired["ai_prompt"] = "Vertical Ancient Rome stone washing basin in a fullonica."
        responses = [
            (
                {
                    "valid": False,
                    "issues": [{"shot_id": 1, "conflicts": ["物件不够具体"]}],
                    "shots": [repaired],
                },
                {"id": "audit-1"},
            ),
            (
                {"valid": True, "issues": [], "shots": [repaired]},
                {"id": "audit-2"},
            ),
        ]
        with patch(
            "app.deepseek_planner.request_json_object", side_effect=responses
        ) as request:
            result, audit = audit_scene_plan(
                "测试文案",
                plan,
                {"semantic_audit": {"enabled": True, "max_repair_rounds": 2}},
                "test-key",
            )
        self.assertEqual(request.call_count, 2)
        self.assertEqual(audit["status"], "reviewed")
        self.assertEqual(result["shots"][0]["objects"], ["stone washing basin"])


class ComfyUIClientTests(unittest.TestCase):
    def test_remote_comfyui_server_is_rejected(self):
        with self.assertRaises(ComfyUIError):
            validate_server_url({"server_url": "https://example.test:8000"})

    def test_api_workflow_marker_seed_size_and_image_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow_path = root / "workflow.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "__AI_VIDEO_PROMPT__"},
                        },
                        "2": {
                            "class_type": "EmptySD3LatentImage",
                            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                        },
                        "3": {
                            "class_type": "KSampler",
                            "inputs": {"seed": 1, "steps": 8, "sampler_name": "euler"},
                        },
                        "4": {
                            "class_type": "SaveImage",
                            "inputs": {"filename_prefix": "old"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            server, thread = _start_server(_ComfyHandler)
            try:
                image_bytes, metadata = generate_comfyui_image(
                    "A Roman ceramic vessel",
                    {
                        "server_url": f"http://127.0.0.1:{server.server_port}",
                        "workflow_file": str(workflow_path),
                        "width": 768,
                        "height": 1344,
                        "poll_interval_seconds": 0.01,
                    },
                    root,
                    seed=12345,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        prompt = _ComfyHandler.prompt_payload["prompt"]
        self.assertEqual(prompt["1"]["inputs"]["text"], "A Roman ceramic vessel")
        self.assertEqual(prompt["2"]["inputs"]["width"], 768)
        self.assertEqual(prompt["2"]["inputs"]["height"], 1344)
        self.assertEqual(prompt["3"]["inputs"]["seed"], 12345)
        self.assertTrue(prompt["4"]["inputs"]["filename_prefix"].startswith("AI-Video-History/"))
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (768, 1344))
        self.assertEqual(metadata["provider"], "comfyui_local")
        self.assertEqual(metadata["request_id"], "prompt-test")


if __name__ == "__main__":
    unittest.main()
