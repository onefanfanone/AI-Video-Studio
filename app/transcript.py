from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence


HARD_END = "。！？!?；;"
SOFT_END = "，,：:"
PUNCTUATION = HARD_END + SOFT_END + "、…—-“”‘’（）()《》"
_WRAP_TOKENIZER: Any | None = None


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass
class TimedChar:
    text: str
    script_index: int
    start: float
    end: float
    matched: bool
    probability: float


def _is_semantic(char: str) -> bool:
    return char.isalnum() or "\u3400" <= char <= "\u9fff"


def _semantic_units(text: str) -> list[tuple[int, str]]:
    return [
        (index, char.casefold())
        for index, char in enumerate(text)
        if _is_semantic(char)
    ]


def _recognized_chars(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for word in words:
        text = re.sub(r"\s+", "", str(word.get("text", "")))
        semantic = [char for char in text if _is_semantic(char)]
        if not semantic:
            continue
        start = max(0.0, float(word.get("start", 0.0)))
        end = max(start + 0.01, float(word.get("end", start + 0.01)))
        probability = float(word.get("probability", 0.0) or 0.0)
        duration = end - start
        for index, char in enumerate(semantic):
            char_start = start + duration * index / len(semantic)
            char_end = start + duration * (index + 1) / len(semantic)
            result.append(
                {
                    "text": char.casefold(),
                    "start": char_start,
                    "end": char_end,
                    "probability": probability,
                }
            )
    return result


def align_script_to_words(
    script: str,
    words: Sequence[dict[str, Any]],
    *,
    audio_duration: float,
) -> tuple[list[TimedChar], float, str]:
    """Align Whisper text to the canonical script and interpolate rare mismatches."""
    script_units = _semantic_units(script)
    recognized = _recognized_chars(words)
    if not script_units or not recognized:
        raise ValueError("文案或 Whisper 逐词结果为空，无法对齐。")
    matcher = SequenceMatcher(
        None,
        [char for _, char in script_units],
        [item["text"] for item in recognized],
        autojunk=False,
    )
    mapping: dict[int, dict[str, Any]] = {}
    matched_count = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            script_index = script_units[block.a + offset][0]
            mapping[script_index] = recognized[block.b + offset]
            matched_count += 1
    coverage = matched_count / len(script_units)

    semantic_indexes = [index for index, _ in script_units]
    semantic_position = {script_index: position for position, script_index in enumerate(semantic_indexes)}
    anchors = sorted(mapping)

    def interpolated(script_index: int) -> tuple[float, float, float]:
        position = semantic_position[script_index]
        previous = next((item for item in reversed(anchors) if item < script_index), None)
        following = next((item for item in anchors if item > script_index), None)
        previous_time = mapping[previous]["end"] if previous is not None else 0.0
        following_time = mapping[following]["start"] if following is not None else audio_duration
        previous_position = semantic_position[previous] if previous is not None else -1
        following_position = (
            semantic_position[following] if following is not None else len(semantic_indexes)
        )
        slots = max(1, following_position - previous_position - 1)
        slot = position - previous_position - 1
        available = max(0.02 * slots, following_time - previous_time)
        start = previous_time + available * slot / slots
        end = previous_time + available * (slot + 1) / slots
        if following is not None:
            end = min(end, following_time)
        return start, max(start + 0.01, end), 0.0

    semantic_timing: dict[int, tuple[float, float, bool, float]] = {}
    for script_index in semantic_indexes:
        if script_index in mapping:
            item = mapping[script_index]
            semantic_timing[script_index] = (
                float(item["start"]),
                float(item["end"]),
                True,
                float(item["probability"]),
            )
        else:
            start, end, probability = interpolated(script_index)
            semantic_timing[script_index] = (start, end, False, probability)

    timed: list[TimedChar] = []
    last_end = 0.0
    for index, char in enumerate(script):
        if char.isspace():
            continue
        if index in semantic_timing:
            start, end, matched, probability = semantic_timing[index]
        else:
            previous = next(
                (semantic_timing[item] for item in reversed(semantic_indexes) if item < index),
                None,
            )
            following = next(
                (semantic_timing[item] for item in semantic_indexes if item > index),
                None,
            )
            start = previous[1] if previous else 0.0
            end = following[0] if following else start + 0.03
            end = min(max(start + 0.01, end), start + 0.08)
            matched = True
            probability = 1.0
        start = max(last_end, min(start, audio_duration))
        end = max(start + 0.01, min(max(end, start + 0.01), audio_duration))
        timed.append(TimedChar(char, index, start, end, matched, probability))
        last_end = end

    recognized_text = "".join(item["text"] for item in recognized)
    return timed, coverage, recognized_text


def wrap_caption(text: str, max_chars_per_line: int, max_lines: int) -> str:
    global _WRAP_TOKENIZER
    text = re.sub(r"\s+", "", text)
    capacity = max_chars_per_line * max_lines
    text = text[:capacity]
    if len(text) <= max_chars_per_line:
        return text
    import jieba

    jieba.setLogLevel(logging.ERROR)
    if _WRAP_TOKENIZER is None:
        _WRAP_TOKENIZER = jieba.Tokenizer()
    minimum_break = max(1, len(text) - max_chars_per_line)
    maximum_break = min(max_chars_per_line, len(text) - 1)
    target = max(minimum_break, min(maximum_break, (len(text) + 1) // 2))
    boundaries: list[int] = []
    cursor = 0
    for token in _WRAP_TOKENIZER.cut(text, HMM=False):
        cursor += len(token)
        if minimum_break <= cursor <= maximum_break:
            boundaries.append(cursor)
    if not boundaries:
        boundaries = list(range(minimum_break, maximum_break + 1))

    def boundary_score(position: int) -> tuple[float, int]:
        score = abs(position - target)
        if text[position - 1] in HARD_END + SOFT_END:
            score -= 2.0
        return score, position

    break_at = min(boundaries, key=boundary_score)
    return text[:break_at] + "\n" + text[break_at : break_at + max_chars_per_line]


def _cue_payload(chars: Sequence[TimedChar], max_chars: int, max_lines: int) -> dict[str, Any]:
    return {
        "start": round(chars[0].start, 6),
        "end": round(chars[-1].end, 6),
        "text": wrap_caption("".join(item.text for item in chars), max_chars, max_lines),
        "script_start": chars[0].script_index,
        "script_end": chars[-1].script_index + 1,
        "emphasis": None,
    }


def build_caption_cues(
    chars: Sequence[TimedChar], subtitle_config: dict[str, Any]
) -> list[dict[str, Any]]:
    max_chars = int(subtitle_config.get("max_chars_per_line", 12))
    max_lines = int(subtitle_config.get("max_lines", 2))
    capacity = max_chars * max_lines
    target_chars = min(capacity, int(subtitle_config.get("target_chars_per_cue", 16)))
    min_duration = float(subtitle_config.get("min_cue_seconds", 0.8))
    max_duration = float(subtitle_config.get("max_cue_seconds", 2.8))
    preserve_short_punctuation_cues = bool(
        subtitle_config.get("preserve_short_punctuation_cues", False)
    )
    minimum_short_cue_chars = int(
        subtitle_config.get("minimum_short_cue_chars", 2)
    )

    if not chars:
        return []

    # jieba keeps common Chinese words such as “征税”“金币”“鼻子” intact. This
    # prevents time-based splitting from producing tails such as “征 / 税。” or
    # “鼻 / 子前”. A local tokenizer avoids leaking project-specific words into
    # later builds in the same Python process.
    import jieba

    jieba.setLogLevel(logging.ERROR)
    tokenizer = jieba.Tokenizer()
    emphasis = subtitle_config.get("emphasis", {})
    custom_words = [
        str(item)
        for key in ("include", "proper_nouns")
        for item in emphasis.get(key, [])
        if str(item)
    ]
    for word in custom_words:
        tokenizer.add_word(word, freq=10_000_000)

    visible_text = "".join(item.text for item in chars)
    tokens = [token for token in tokenizer.cut(visible_text, HMM=False) if token]
    units: list[list[TimedChar]] = []
    cursor = 0
    for token in tokens:
        token_length = len(token)
        unit = list(chars[cursor : cursor + token_length])
        if not unit or "".join(item.text for item in unit) != token:
            units = [[item] for item in chars]
            break
        units.append(unit)
        cursor += token_length
    if cursor != len(chars):
        units = [[item] for item in chars]

    # Keep commas, colons and sentence marks with the preceding word. Treating
    # punctuation as an independent unit can produce visually broken cards
    # such as "，皇帝韦斯巴芗" when a one-line cue reaches its capacity.
    punctuation_merged: list[list[TimedChar]] = []
    for unit in units:
        if punctuation_merged and not any(_is_semantic(item.text) for item in unit):
            punctuation_merged[-1].extend(unit)
        else:
            punctuation_merged.append(unit)
    units = punctuation_merged

    def flatten(group: Sequence[Sequence[TimedChar]]) -> list[TimedChar]:
        return [item for unit in group for item in unit]

    def group_duration(group: Sequence[Sequence[TimedChar]]) -> float:
        flat = flatten(group)
        return flat[-1].end - flat[0].start

    def group_chars(group: Sequence[Sequence[TimedChar]]) -> int:
        return sum(len(unit) for unit in group)

    groups: list[list[list[TimedChar]]] = []
    current: list[list[TimedChar]] = []
    for unit in units:
        projected = current + [unit]
        if current and (
            group_chars(projected) > capacity
            or group_duration(projected) > max_duration + 0.02
        ):
            groups.append(current)
            current = []
        current.append(unit)
        duration = group_duration(current)
        text = "".join(item.text for item in unit)
        if text[-1] in HARD_END:
            groups.append(current)
            current = []
        elif text[-1] in SOFT_END and (
            group_chars(current) >= target_chars or duration >= min_duration
        ):
            groups.append(current)
            current = []
        elif group_chars(current) >= target_chars and duration >= max_duration * 0.72:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    # If a long clause produced two uneven cues, move the boundary only across
    # whole jieba tokens. Prefer balanced durations and punctuation boundaries.
    for index in range(len(groups) - 1):
        left_group = groups[index]
        right_group = groups[index + 1]
        left_flat = flatten(left_group)
        if left_flat[-1].text in HARD_END + SOFT_END:
            continue
        combined = left_group + right_group
        current_score = abs(group_duration(left_group) - group_duration(right_group))
        best: tuple[float, int] | None = None
        for split in range(1, len(combined)):
            candidate_left = combined[:split]
            candidate_right = combined[split:]
            left_duration = group_duration(candidate_left)
            right_duration = group_duration(candidate_right)
            if not (
                min_duration - 0.02 <= left_duration <= max_duration + 0.02
                and min_duration - 0.02 <= right_duration <= max_duration + 0.02
                and group_chars(candidate_left) <= capacity
                and group_chars(candidate_right) <= capacity
            ):
                continue
            score = abs(left_duration - right_duration)
            candidate_text = "".join(item.text for item in flatten(candidate_left))
            if candidate_text[-1] in HARD_END + SOFT_END:
                score -= 0.5
            if best is None or score < best[0]:
                best = (score, split)
        if best and (
            min(group_duration(left_group), group_duration(right_group)) < min_duration
            or best[0] + 0.15 < current_score
        ):
            groups[index] = combined[: best[1]]
            groups[index + 1] = combined[best[1] :]

    # Merge any remaining short clause with a neighbour when it stays inside
    # the configured duration and two-line capacity.
    index = 0
    while index < len(groups):
        if group_duration(groups[index]) >= min_duration - 0.02:
            index += 1
            continue
        group_text = "".join(item.text for item in flatten(groups[index]))
        semantic_count = sum(1 for char in group_text if _is_semantic(char))
        if (
            preserve_short_punctuation_cues
            and semantic_count >= minimum_short_cue_chars
            and group_text[-1] in HARD_END + SOFT_END
        ):
            index += 1
            continue
        merged = False
        if index + 1 < len(groups):
            candidate = groups[index] + groups[index + 1]
            if group_chars(candidate) <= capacity and group_duration(candidate) <= max_duration + 0.02:
                groups[index : index + 2] = [candidate]
                merged = True
        if not merged and index > 0:
            candidate = groups[index - 1] + groups[index]
            if group_chars(candidate) <= capacity and group_duration(candidate) <= max_duration + 0.02:
                groups[index - 1 : index + 1] = [candidate]
                index -= 1
                merged = True
        if not merged:
            index += 1

    merged_chars = [flatten(group) for group in groups if group]
    result = [_cue_payload(group, max_chars, max_lines) for group in merged_chars]
    for left, right in zip(result, result[1:]):
        left["end"] = min(left["end"], right["start"])
    return result


def _automatic_candidates(text: str, proper_nouns: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    candidates.extend(
        match
        for groups in re.findall(r"“([^”]{1,6})”|‘([^’]{1,6})’", text)
        for match in groups
        if match
    )
    candidates.extend(re.findall(r"\d+(?:\.\d+)?(?:万|亿|年|次|秒|%)?", text))
    candidates.extend(noun for noun in proper_nouns if noun in text)
    for trigger in ("更离谱的是", "其实", "但", "于是", "竟然"):
        if trigger in text:
            tail = text.split(trigger, 1)[1]
            match = re.search(r"[\u3400-\u9fff]{2,4}", tail)
            if match:
                candidates.append(match.group(0))
    return candidates


def apply_emphasis(
    script: str,
    chars: Sequence[TimedChar],
    cues: list[dict[str, Any]],
    emphasis_config: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(emphasis_config.get("mode", "hybrid"))
    included = [str(item) for item in emphasis_config.get("include", []) if str(item)]
    excluded = {str(item) for item in emphasis_config.get("exclude", [])}
    proper_nouns = [str(item) for item in emphasis_config.get("proper_nouns", [])]
    by_script_index = {item.script_index: item for item in chars}
    events: list[dict[str, Any]] = []

    for cue_index, cue in enumerate(cues):
        flat = cue["text"].replace("\n", "")
        candidates: list[str] = []
        if mode in {"hybrid", "manual"}:
            candidates.extend(phrase for phrase in included if phrase in flat)
        if mode in {"hybrid", "automatic", "auto"}:
            candidates.extend(_automatic_candidates(flat, proper_nouns))
        phrase = next(
            (
                candidate
                for candidate in candidates
                if candidate not in excluded
                and candidate in flat
                and 1 <= len(candidate) <= 6
                and len(candidate) <= max(1, round(len(flat) * 0.4))
            ),
            None,
        )
        if not phrase:
            continue
        local_start = flat.index(phrase)
        script_slice = script[cue["script_start"] : cue["script_end"]]
        compact_positions = [
            cue["script_start"] + index
            for index, char in enumerate(script_slice)
            if not char.isspace()
        ]
        selected_positions = compact_positions[local_start : local_start + len(phrase)]
        selected_chars = [
            by_script_index[position]
            for position in selected_positions
            if position in by_script_index and _is_semantic(script[position])
        ]
        if not selected_chars:
            continue
        event = {
            "cue_index": cue_index,
            "phrase": phrase,
            "text_start": local_start,
            "text_end": local_start + len(phrase),
            "start": round(selected_chars[0].start, 6),
            "end": round(min(cue["end"], selected_chars[-1].end + 0.12), 6),
        }
        cue["emphasis"] = event
        events.append(event)
    return events


def build_transcript(
    script: str,
    alignment: dict[str, Any],
    *,
    audio_duration: float,
    subtitle_config: dict[str, Any],
) -> tuple[dict[str, Any], list[Cue], list[dict[str, Any]]]:
    chars, coverage, recognized_text = align_script_to_words(
        script, alignment.get("words", []), audio_duration=audio_duration
    )
    minimum_coverage = float(subtitle_config.get("minimum_alignment_coverage", 0.98))
    if coverage + 1e-9 < minimum_coverage:
        raise ValueError(
            f"Whisper 与原文匹配覆盖率只有 {coverage:.2%}，低于要求的 {minimum_coverage:.2%}。"
        )
    cue_payloads = build_caption_cues(chars, subtitle_config)
    events = apply_emphasis(
        script,
        chars,
        cue_payloads,
        subtitle_config.get("emphasis", {}),
    )
    transcript = {
        "schema_version": 1,
        "normalized_script": script,
        "audio_duration": audio_duration,
        "alignment": {
            "provider": alignment.get("engine", "faster-whisper"),
            "model": alignment.get("model"),
            "device": alignment.get("device"),
            "coverage": coverage,
            "minimum_coverage": minimum_coverage,
            "recognized_text": recognized_text,
        },
        "limits": {
            "max_chars_per_line": int(subtitle_config.get("max_chars_per_line", 12)),
            "max_lines": int(subtitle_config.get("max_lines", 2)),
            "min_cue_seconds": float(subtitle_config.get("min_cue_seconds", 0.8)),
            "max_cue_seconds": float(subtitle_config.get("max_cue_seconds", 2.8)),
        },
        "words": [asdict(item) for item in chars],
        "cues": cue_payloads,
    }
    cues = [Cue(item["start"], item["end"], item["text"]) for item in cue_payloads]
    return transcript, cues, events


def cues_from_transcript(transcript: dict[str, Any]) -> list[Cue]:
    return [
        Cue(float(item["start"]), float(item["end"]), str(item["text"]))
        for item in transcript.get("cues", [])
    ]
