from __future__ import annotations

import re
from typing import Any


TIME_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string"},
        "region": {"type": "string"},
        "start_year": {"type": ["integer", "null"]},
        "end_year": {"type": ["integer", "null"]},
        "confidence": {
            "type": "string",
            "enum": ["high", "approximate", "unknown"],
        },
        "period_terms_en": {"type": "array", "items": {"type": "string"}},
        "period_terms_zh": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "label",
        "region",
        "start_year",
        "end_year",
        "confidence",
        "period_terms_en",
        "period_terms_zh",
    ],
}

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent_id": {"type": "string"},
                    "shot_id": {"type": "integer"},
                    "era": {"type": "string"},
                    "time_context": TIME_CONTEXT_SCHEMA,
                    "location": {"type": "string"},
                    "people": {"type": "array", "items": {"type": "string"}},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "action": {"type": "string"},
                    "mood": {"type": "string"},
                    "search_terms_zh": {"type": "array", "items": {"type": "string"}},
                    "search_terms_en": {"type": "array", "items": {"type": "string"}},
                    "must_include": {"type": "array", "items": {"type": "string"}},
                    "avoid": {"type": "array", "items": {"type": "string"}},
                    "ai_prompt": {"type": "string"},
                },
                "required": [
                    "intent_id",
                    "shot_id",
                    "era",
                    "time_context",
                    "location",
                    "people",
                    "objects",
                    "action",
                    "mood",
                    "search_terms_zh",
                    "search_terms_en",
                    "must_include",
                    "avoid",
                    "ai_prompt",
                ],
            },
        }
    },
    "required": ["shots"],
}

SHOT_REQUIRED = tuple(SCENE_SCHEMA["properties"]["shots"]["items"]["required"])
STRING_FIELDS = ("intent_id", "era", "location", "action", "mood", "ai_prompt")
STRING_LIST_FIELDS = (
    "people",
    "objects",
    "search_terms_zh",
    "search_terms_en",
    "must_include",
    "avoid",
)
TIME_REQUIRED = tuple(TIME_CONTEXT_SCHEMA["required"])
GENERIC_WORDS = {
    "object",
    "objects",
    "element",
    "elements",
    "item",
    "items",
    "thing",
    "things",
    "scene",
    "visual",
    "historical",
    "period",
}
ENGLISH_NEGATION_PREFIX = re.compile(
    r"(?:\b(?:no|not|without|avoid(?:ing)?|exclude(?:ing)?|free of)"
    r"|\b(?:do not|must not)\s+(?:show|include|depict|feature|contain))"
    r"(?:\s+[\w-]+){0,5}\s*$",
    re.IGNORECASE,
)
CHINESE_NEGATION_PREFIX = re.compile(
    r"(?:不要|不得|避免|排除|不应|不能|不出现|没有|无)[^，。；,.!?！？]{0,16}$"
)
ENGLISH_LEADING_NEGATION = re.compile(
    r"^\s*(?:(?:do not|must not)\s+(?:show|include|depict|feature|contain)\s+|"
    r"(?:no|not|without|avoid(?:ing)?|exclude(?:ing)?|free of)\s+)",
    re.IGNORECASE,
)
CHINESE_LEADING_NEGATION = re.compile(
    r"^\s*(?:不要|不得|避免|排除|不应|不能|不出现|没有|无)"
    r"(?:出现|包含|展示|描绘|使用)?"
)


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}", str(value))
        if token.lower() not in GENERIC_WORDS
    }


def _markers(value: str) -> set[str]:
    text = str(value or "").strip().lower()
    text = ENGLISH_LEADING_NEGATION.sub("", text)
    text = CHINESE_LEADING_NEGATION.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:，。；：")
    # An avoid entry is a phrase-level constraint. Splitting "paper money" into
    # "paper" and "money", or "modern washing machines" into "washing", causes
    # obvious false positives in historically relevant descriptions. Semantic
    # synonym detection belongs to the DeepSeek audit; local code only rejects an
    # unambiguous affirmative occurrence of the complete forbidden phrase.
    return {text} if text and text not in GENERIC_WORDS else set()


def _affirmative_occurrence(text: str, marker: str) -> bool:
    normalized = str(text or "").lower()
    for match in re.finditer(re.escape(marker.lower()), normalized):
        prefix = normalized[max(0, match.start() - 72) : match.start()]
        if ENGLISH_NEGATION_PREFIX.search(prefix) or CHINESE_NEGATION_PREFIX.search(prefix):
            continue
        return True
    return False


def validate_plan_shots(plan: Any, expected_shots: int) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or set(plan) != {"shots"}:
        raise ValueError("顶层必须只包含 shots。")
    shots = plan.get("shots")
    if not isinstance(shots, list) or len(shots) != expected_shots:
        raise ValueError(f"镜头数必须为 {expected_shots}。")
    required = set(SHOT_REQUIRED)
    validated: list[dict[str, Any]] = []
    for index, intent in enumerate(shots, 1):
        if not isinstance(intent, dict) or set(intent) != required:
            raise ValueError(f"镜头 {index} 的字段不符合 scene-plan-v3。")
        if not isinstance(intent.get("shot_id"), int) or isinstance(intent.get("shot_id"), bool):
            raise ValueError(f"镜头 {index} 的 shot_id 必须为整数。")
        for field in STRING_FIELDS:
            if not isinstance(intent.get(field), str) or not intent[field].strip():
                raise ValueError(f"镜头 {index} 的 {field} 必须为非空字符串。")
        for field in STRING_LIST_FIELDS:
            value = intent.get(field)
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"镜头 {index} 的 {field} 必须为非空字符串数组。")
        if not intent["search_terms_en"]:
            raise ValueError(f"镜头 {index} 至少需要一个英文馆藏搜索词。")
        if not intent["must_include"]:
            raise ValueError(f"镜头 {index} 至少需要一个 must_include。")
        if not intent["avoid"]:
            raise ValueError(f"镜头 {index} 至少需要一个动态 avoid 元素。")

        context = intent.get("time_context")
        if not isinstance(context, dict) or set(context) != set(TIME_REQUIRED):
            raise ValueError(f"镜头 {index} 的 time_context 字段不完整。")
        for field in ("label", "region", "confidence"):
            if not isinstance(context.get(field), str) or not context[field].strip():
                raise ValueError(f"镜头 {index} 的 time_context.{field} 必须为非空字符串。")
        if context["confidence"] not in {"high", "approximate", "unknown"}:
            raise ValueError(f"镜头 {index} 的 time_context.confidence 不合法。")
        for field in ("period_terms_en", "period_terms_zh"):
            value = context.get(field)
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"镜头 {index} 的 time_context.{field} 必须为字符串数组。")
        if not context["period_terms_en"]:
            raise ValueError(f"镜头 {index} 至少需要一个英文时代锚点。")
        for field in ("start_year", "end_year"):
            value = context.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"镜头 {index} 的 time_context.{field} 必须为整数或 null。")
        if (
            context["start_year"] is not None
            and context["end_year"] is not None
            and context["start_year"] > context["end_year"]
        ):
            raise ValueError(f"镜头 {index} 的时代起始年份晚于结束年份。")

        required_phrases = [
            re.sub(r"\s+", " ", item.lower()).strip(" ,.;:，。；：")
            for item in intent["must_include"]
        ]
        forbidden_phrases = [
            re.sub(
                r"\s+",
                " ",
                CHINESE_LEADING_NEGATION.sub(
                    "", ENGLISH_LEADING_NEGATION.sub("", item.lower())
                ),
            ).strip(" ,.;:，。；：")
            for item in intent["avoid"]
        ]
        direct_overlap = next(
            (
                forbidden
                for forbidden in forbidden_phrases
                if forbidden
                and any(
                    required == forbidden
                    or re.search(rf"(?<!\w){re.escape(forbidden)}(?!\w)", required)
                    for required in required_phrases
                )
            ),
            None,
        )
        if direct_overlap:
            raise ValueError(
                f"镜头 {index} 的 must_include 与 avoid 冲突：{direct_overlap}。"
            )

        positive_context = " ".join(
            [
                intent["location"],
                *intent["people"],
                *intent["objects"],
                intent["action"],
                *intent["search_terms_zh"],
                *intent["search_terms_en"],
                *intent["must_include"],
            ]
        )
        for forbidden in intent["avoid"]:
            for marker in sorted(_markers(forbidden), key=len, reverse=True):
                if _affirmative_occurrence(positive_context, marker) or _affirmative_occurrence(
                    intent["ai_prompt"], marker
                ):
                    raise ValueError(
                        f"镜头 {index} 把禁用元素“{forbidden}”写成了肯定画面要求。"
                    )

        search_text = " ".join(intent["search_terms_en"]).lower()
        period_tokens = _tokens(" ".join(context["period_terms_en"]))
        subject_tokens = _tokens(
            " ".join([*intent["objects"], *intent["must_include"], intent["action"]])
        )
        if period_tokens and not (period_tokens & _tokens(search_text)):
            raise ValueError(f"镜头 {index} 的英文搜索词缺少时代/文化锚点。")
        if subject_tokens and not (subject_tokens & _tokens(search_text)):
            raise ValueError(f"镜头 {index} 的英文搜索词缺少具体物件或工艺锚点。")
        prompt_tokens = _tokens(intent["ai_prompt"])
        if period_tokens and not (period_tokens & prompt_tokens):
            raise ValueError(f"镜头 {index} 的 AI 提示词缺少时代/文化锚点。")
        validated.append(dict(intent))
    return validated
