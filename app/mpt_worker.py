from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


async def _tts(request: dict[str, Any]) -> dict[str, Any]:
    import edge_tts

    output_audio = Path(request["output_audio"])
    output_metadata = Path(request["output_metadata"])
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    communicator = edge_tts.Communicate(
        text=request["text"],
        voice=request["voice"],
        rate=request.get("rate", "+0%"),
        pitch=request.get("pitch", "+0Hz"),
        boundary="WordBoundary",
    )
    await communicator.save(str(output_audio), str(output_metadata))
    return {
        "audio": str(output_audio),
        "metadata": str(output_metadata),
        "engine": "MoneyPrinterTurbo isolated Edge TTS",
    }


def _align(request: dict[str, Any]) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    model_name = request.get("model", "large-v3")
    device = request.get("device", "cuda")
    compute_type = request.get("compute_type", "float16" if device == "cuda" else "int8")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=request.get("download_root"),
    )
    segments, info = model.transcribe(
        request["audio"],
        language="zh",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        initial_prompt=str(request.get("initial_prompt") or "") or None,
    )
    words: list[dict[str, Any]] = []
    segment_count = 0
    for segment in segments:
        segment_count += 1
        for word in segment.words or []:
            text = str(word.word).strip()
            if not text:
                continue
            words.append(
                {
                    "text": text,
                    "start": round(float(word.start), 6),
                    "end": round(float(word.end), 6),
                    "probability": round(float(getattr(word, "probability", 0.0) or 0.0), 6),
                }
            )
    if not words:
        raise RuntimeError("Whisper 没有识别出任何逐词时间戳。")
    output_alignment = Path(request["output_alignment"])
    payload = {
        "schema_version": 1,
        "engine": "MoneyPrinterTurbo faster-whisper",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segment_count": segment_count,
        "words": words,
    }
    _write_json(output_alignment, payload)
    return {"alignment": str(output_alignment), "word_count": len(words)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        operation = request.get("operation")
        if operation == "tts":
            result = asyncio.run(_tts(request))
        elif operation == "align":
            result = _align(request)
        else:
            raise ValueError(f"不支持的 worker 操作：{operation}")
        _write_json(
            args.response,
            {
                "status": "success",
                "operation": operation,
                "moneyprinterturbo": request.get("moneyprinterturbo"),
                **result,
            },
        )
        return 0
    except Exception as exc:
        _write_json(
            args.response,
            {
                "status": "failed",
                "operation": request.get("operation"),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
