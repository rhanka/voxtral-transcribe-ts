# Benchmark: voxtral-transcribe-ts vs faster-whisper

This benchmark harness compares:

- recognition quality: `WER` and `CER`
- speed: total transcription time and real-time factor, `RTF = processing duration / audio duration`
- runtime profile: model load time and peak RSS when available

The package itself remains Node/TypeScript-only. Python is only used by the optional `faster-whisper` comparison runner under `bench/`.

## Dataset Manifest

Create a JSONL manifest where each line describes one audio sample:

```json
{"id":"fr_001","audio":"./audio/fr_001.wav","text":"bonjour je veux tester la transcription","language":"fr","durationMs":4200}
```

Fields:

- `id`: stable sample id
- `audio`: path relative to the manifest file, or absolute path
- `text`: reference transcript
- `language`: optional language hint
- `durationMs`: optional audio duration; WAV duration is inferred when omitted

See `bench/manifest.example.jsonl` for the expected shape.

## Voxtral Runner

Build the package first because the runner imports the local `dist` entry:

```bash
npm run build
npm run bench:voxtral -- \
  --manifest bench/datasets/manifest.jsonl \
  --out bench/results/voxtral.jsonl \
  --decoder ffmpeg \
  --mode warm
```

Enterprise/local-model profile:

```bash
npm run bench:voxtral -- \
  --manifest bench/datasets/manifest.jsonl \
  --out bench/results/voxtral-local-model.jsonl \
  --decoder ffmpeg \
  --model-path /opt/models/Voxtral-Mini-4B-Realtime-2602-ONNX \
  --require-local-model \
  --mode warm
```

## faster-whisper Runner

Install `faster-whisper` in a separate benchmark environment:

```bash
python3 -m venv .venv-bench
. .venv-bench/bin/activate
pip install faster-whisper
```

Run a CPU/int8 profile:

```bash
npm run bench:faster-whisper -- \
  --manifest bench/datasets/manifest.jsonl \
  --out bench/results/faster-whisper-small-int8.jsonl \
  --model-size small \
  --device cpu \
  --compute-type int8 \
  --mode warm
```

## Score Results

```bash
npm run bench:score -- \
  --inputs bench/results/voxtral.jsonl bench/results/faster-whisper-small-int8.jsonl \
  --out bench/results/summary.json
```

The scorer prints a Markdown table and writes a JSON summary.

## Fairness Rules

- Report both `cold` and `warm` runs when startup latency matters.
- Use the same manifest, same normalized references, and same hardware.
- Keep model profiles explicit: for example `voxtral q4 cpu` vs `faster-whisper small int8 cpu`.
- Do not compare a browser run against a server Python run unless that is the product question.
- Treat model download/provisioning separately from transcription speed.

## Current Status

- Harness exists.
- No benchmark result is claimed yet.
- The next step is to run it on a real French corpus and commit the generated summary separately if we want published numbers.
