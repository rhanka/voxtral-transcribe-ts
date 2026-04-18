#!/usr/bin/env python3

import argparse
import json
import shutil
import wave
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a FLEURS manifest for ASR WER/CER benchmarking.")
    parser.add_argument("--config", default="fr_fr", help="FLEURS config, for example fr_fr or en_us")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="bench/datasets/fleurs-fr-test.jsonl")
    parser.add_argument("--audio-dir", default="bench/datasets/audio/fleurs-fr-test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-source-format", action="store_true")
    return parser.parse_args()


def duration_ms(path):
    if path.suffix.lower() != ".wav":
        return None
    with wave.open(str(path), "rb") as wav:
        return int((wav.getnframes() / wav.getframerate()) * 1000)


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Missing dependency: pip install datasets soundfile") from error

    dataset = load_dataset(
        "google/fleurs",
        args.config,
        split=args.split,
        trust_remote_code=True,
    )
    out_path = Path(args.out).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, row in enumerate(dataset):
        if args.limit is not None and index >= args.limit:
            break

        source_path = Path(row["path"])
        suffix = source_path.suffix if args.keep_source_format else ".wav"
        audio_path = audio_dir / f"{args.config}_{args.split}_{index:05d}{suffix}"

        if args.keep_source_format:
            shutil.copyfile(source_path, audio_path)
        else:
            import soundfile as sf

            audio = row["audio"]
            sf.write(audio_path, audio["array"], audio["sampling_rate"])

        record = {
            "id": f"{args.config}_{args.split}_{index:05d}",
            "audio": normalize_manifest_path(audio_path, out_path.parent),
            "text": row["transcription"],
            "language": args.config.split("_")[0],
            "durationMs": duration_ms(audio_path),
            "benchmarkPurpose": "wer",
            "dataset": "google/fleurs",
            "config": args.config,
            "split": args.split,
        }

        rows.append(record)

    with out_path.open("w", encoding="utf-8") as out:
        for record in rows:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} FLEURS records to {out_path}")
    print(f"Audio directory: {audio_dir}")


def normalize_manifest_path(path, manifest_dir):
    try:
        rel = path.relative_to(manifest_dir)
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
