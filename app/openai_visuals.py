from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .scene_plan_schema import SCENE_SCHEMA, validate_plan_shots
from .studio_settings import load_runtime_secrets


class OpenAIVisualError(RuntimeError):
    """A redacted, user-facing error from the optional OpenAI visual layer."""


def load_local_env(root: Path) -> dict[str, str]:
    """Load DPAPI credentials, with legacy .env.local compatibility."""
    return load_runtime_secrets(root)


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Video/3.0",
        },
        method="POST",
    )
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
                if response.headers.get("x-request-id"):
                    result.setdefault("_request_id", response.headers["x-request-id"])
                return result
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))
    lowered = last_error.lower()
    if any(marker in lowered for marker in ("moderation", "safety", "content_policy")):
        raise OpenAIVisualError(
            "OpenAI safety/moderation 拒绝了这张候选图。请在审核阶段人工补图；"
            "程序不会弱化或绕过安全要求。"
        )
    raise OpenAIVisualError(f"OpenAI API 请求失败（已重试两次）：{last_error}")


def create_scene_plan(
    script: str,
    shots: list[dict[str, Any]],
    planner: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise OpenAIVisualError(
            "sourced 模式需要 OPENAI_API_KEY。请把它写入项目根目录 .env.local；"
            "密钥不会进入任务文件或报告。"
        )
    shot_inputs = [
        {
            "shot_id": index + 1,
            "start": shot["start"],
            "end": shot["end"],
            "narration": shot.get("caption_text", ""),
        }
        for index, shot in enumerate(shots)
    ]
    prompt = (
        "你是历史短视频的画面资料编辑。只把给定文案和整数帧镜头转换为检索意图；"
        "不得改写文案、补充文案没有陈述的历史事实或断言。每个镜头必须对应一个 intent。"
        "搜索词优先使用博物馆馆藏常用的物件、地点、时代英文名称。AI 提示词要写成竖屏电影感"
        "写实历史重构。每镜头必须给出 time_context、动态 must_include 和与该时代相关的具体"
        "avoid；不得依赖预设文明或现代物品词表。搜索词必须同时包含时代/文化锚点和具体物件/工艺。"
        "AI 提示词明确年代、材质、服饰、构图和底部字幕安全区。\n\n"
        + json.dumps({"script": script, "shots": shot_inputs}, ensure_ascii=False)
    )
    payload = {
        "model": planner.get("model", "gpt-5.6-luna"),
        "store": False,
        "reasoning": {"effort": planner.get("reasoning_effort", "low")},
        "input": [{"role": "user", "content": prompt}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scene_plan",
                "strict": True,
                "schema": SCENE_SCHEMA,
            }
        },
    }
    endpoint = str(planner.get("base_url", "https://api.openai.com/v1")).rstrip(
        "/"
    ) + "/responses"
    raw = _post_json(endpoint, payload, api_key)
    output_text = ""
    for output in raw.get("output", []):
        for item in output.get("content", []):
            if item.get("type") == "output_text":
                output_text += str(item.get("text", ""))
    try:
        plan = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise OpenAIVisualError("OpenAI 视觉计划未返回有效的结构化 JSON。") from exc
    try:
        intents = validate_plan_shots(plan, len(shots))
    except ValueError as exc:
        raise OpenAIVisualError(f"OpenAI 视觉计划未通过本地校验：{exc}") from exc
    for index, intent in enumerate(intents, 1):
        intent["shot_id"] = index
        intent["intent_id"] = f"intent-{index:03d}"
        intent["start"] = shots[index - 1]["start"]
        intent["end"] = shots[index - 1]["end"]
        intent["narration"] = shots[index - 1].get("caption_text", "")
    result = {
        "schema_version": 2,
        "provider": "openai",
        "model": planner.get("model", "gpt-5.6-luna"),
        "prompt_version": planner.get("prompt_version", "scene-plan-v3"),
        "response_id": raw.get("id"),
        "shots": intents,
    }
    safe_raw = {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "created_at": raw.get("created_at"),
        "usage": raw.get("usage"),
        "status": raw.get("status"),
    }
    return result, safe_raw


def generate_image(
    prompt: str,
    config: dict[str, Any],
    api_key: str,
) -> tuple[bytes, dict[str, Any]]:
    if not api_key:
        raise OpenAIVisualError("AI 缺图兜底需要 OPENAI_API_KEY。")
    payload = {
        "model": config.get("model", "gpt-image-2"),
        "prompt": prompt,
        "size": config.get("size", "1024x1536"),
        "quality": config.get("quality", "medium"),
        "output_format": config.get("output_format", "jpeg"),
        "n": 1,
    }
    endpoint = str(config.get("base_url", "https://api.openai.com/v1")).rstrip(
        "/"
    ) + "/images/generations"
    raw = _post_json(endpoint, payload, api_key)
    data = raw.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise OpenAIVisualError(
            "OpenAI 未返回图片，可能是 moderation 拒绝或请求参数不可用；"
            "程序不会尝试绕过安全限制。"
        )
    image_bytes = base64.b64decode(data[0]["b64_json"], validate=True)
    metadata = {
        "provider": "openai",
        "model": config.get("model", "gpt-image-2"),
        "request_id": raw.get("id") or raw.get("_request_id"),
        "prompt": prompt,
        "size": config.get("size", "1024x1536"),
        "quality": config.get("quality", "medium"),
        "output_format": config.get("output_format", "jpeg"),
        "generated_at": int(time.time()),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    return image_bytes, metadata
