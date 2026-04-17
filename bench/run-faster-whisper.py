#!/usr/bin/env python3

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run faster-whisper benchmark records.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="bench/results/faster-whisper.jsonl")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language")
    parser.add_argument("--mode", choices=["warm", "cold"], default="warm")
    return parser.parse_args()


def read_manifest(path):
    manifest_path = Path(path).resolve()
    items = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "audio" not in item:
                raise ValueError(f"Manifest line {index} must include id and audio")
            audio = Path(item["audio"])
            item["audioPath"] = str(audio if audio.is_absolute() else manifest_path.parent / audio)
            items.append(item)
    return items


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def load_model(args):
    from faster_whisper import WhisperModel

    started_at = time.perf_counter()
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    return model, (time.perf_counter() - started_at) * 1000


def transcribe_one(model, args, item, model_load_ms):
    started_at = time.perf_counter()
    language = item.get("language") or args.language

    segments, info = model.transcribe(item["audioPath"], language=language)
    hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
    duration_ms = (time.perf_counter() - started_at) * 1000
    audio_duration_ms = item.get("durationMs")
    if audio_duration_ms is None and getattr(info, "duration", None):
        audio_duration_ms = info.duration * 1000

    return {
        "engine": "faster-whisper",
        "engineVersion": package_version(),
        "id": item["id"],
        "audio": item["audio"],
        "language": language,
        "reference": item.get("text"),
        "hypothesis": hypothesis,
        "durationMs": duration_ms,
        "audioDurationMs": audio_duration_ms,
        "rtf": duration_ms / audio_duration_ms if audio_duration_ms else None,
        "model": args.model_size,
        "device": args.device,
        "dtype": args.compute_type,
        "decoder": "faster-whisper",
        "mode": args.mode,
        "modelLoadMs": model_load_ms,
        "peakRssBytes": peak_rss_bytes(),
    }


def package_version():
    try:
        from importlib.metadata import version

        return version("faster-whisper")
    except Exception:
        return None


def main():
    args = parse_args()
    items = read_manifest(args.manifest)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warm_model = None
    warm_load_ms = None
    if args.mode == "warm":
        warm_model, warm_load_ms = load_model(args)

    with out_path.open("w", encoding="utf-8") as out:
        for item in items:
            model = warm_model
            model_load_ms = warm_load_ms

            try:
                if args.mode == "cold":
                    model, model_load_ms = load_model(args)
                record = transcribe_one(model, args, item, model_load_ms)
            except Exception as error:
                record = {
                    "engine": "faster-whisper",
                    "engineVersion": package_version(),
                    "id": item.get("id"),
                    "audio": item.get("audio"),
                    "language": item.get("language") or args.language,
                    "reference": item.get("text"),
                    "error": str(error),
                    "mode": args.mode,
                    "modelLoadMs": model_load_ms,
                    "peakRssBytes": peak_rss_bytes(),
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
