#!/bin/bash

set -euo pipefail

if [ -z "${MANIFEST:-}" ]; then
  echo "MANIFEST is required, for example: MANIFEST=bench/datasets/common-voice-fr-test.jsonl npm run bench:suite" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-bench/results/$STAMP}"
VOXTRAL_OUT="$RESULTS_DIR/voxtral.jsonl"
FASTER_WHISPER_OUT="$RESULTS_DIR/faster-whisper.jsonl"
SUMMARY_OUT="$RESULTS_DIR/summary.json"

VOXTRAL_DECODER="${VOXTRAL_DECODER:-ffmpeg}"
VOXTRAL_MODE="${VOXTRAL_MODE:-warm}"
VOXTRAL_DEVICE="${VOXTRAL_DEVICE:-cpu}"
VOXTRAL_DTYPE="${VOXTRAL_DTYPE:-q4}"

FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-small}"
FASTER_WHISPER_DEVICE="${FASTER_WHISPER_DEVICE:-cpu}"
FASTER_WHISPER_COMPUTE_TYPE="${FASTER_WHISPER_COMPUTE_TYPE:-int8}"
FASTER_WHISPER_MODE="${FASTER_WHISPER_MODE:-warm}"

mkdir -p "$RESULTS_DIR"

echo "Benchmark manifest: $MANIFEST"
echo "Results directory: $RESULTS_DIR"
echo ""

npm run build

VOXTRAL_ARGS=(
  --manifest "$MANIFEST"
  --out "$VOXTRAL_OUT"
  --decoder "$VOXTRAL_DECODER"
  --mode "$VOXTRAL_MODE"
  --device "$VOXTRAL_DEVICE"
  --dtype "$VOXTRAL_DTYPE"
)

if [ -n "${VOXTRAL_MODEL_PATH:-}" ]; then
  VOXTRAL_ARGS+=(--model-path "$VOXTRAL_MODEL_PATH" --require-local-model)
fi

if [ -n "${VOXTRAL_CACHE_DIR:-}" ]; then
  VOXTRAL_ARGS+=(--cache-dir "$VOXTRAL_CACHE_DIR")
fi

npm run bench:voxtral -- "${VOXTRAL_ARGS[@]}"

npm run bench:faster-whisper -- \
  --manifest "$MANIFEST" \
  --out "$FASTER_WHISPER_OUT" \
  --model-size "$FASTER_WHISPER_MODEL" \
  --device "$FASTER_WHISPER_DEVICE" \
  --compute-type "$FASTER_WHISPER_COMPUTE_TYPE" \
  --mode "$FASTER_WHISPER_MODE"

npm run bench:score -- \
  --inputs "$VOXTRAL_OUT" "$FASTER_WHISPER_OUT" \
  --out "$SUMMARY_OUT"

echo ""
echo "Summary: $SUMMARY_OUT"
