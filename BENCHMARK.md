# Benchmark: voxtral-transcribe-ts vs faster-whisper

This benchmark harness compares:

- recognition quality: `WER` and `CER`
- speed: total transcription time and real-time factor, `RTF = processing duration / audio duration`
- runtime profile: model load time and peak RSS when available

The package itself remains Node/TypeScript-only. Python is only used by the optional `faster-whisper` comparison runner under `bench/`.

## Protocol

Use two complementary benchmarks:

- `FLEURS fr_fr test` for recognition quality, scored with `WER` and `CER`
- `SYSTRAN/faster-whisper benchmark.m4a` for speed, matching the upstream faster-whisper speed benchmark audio

The speed audio is not used as the main WER benchmark because it is a single long file, not a statistically useful ASR corpus. FLEURS is not the main speed benchmark because it is many short clips and includes dataset iteration overhead.

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

For Common Voice or an internal CSV/TSV corpus, generate a manifest with:

```bash
npm run bench:manifest -- \
  --transcripts bench/datasets/raw/common-voice-fr/test.tsv \
  --audio-dir bench/datasets/raw/common-voice-fr/clips \
  --out bench/datasets/common-voice-fr-test.jsonl \
  --id-column path \
  --audio-column path \
  --text-column sentence \
  --language fr
```

More dataset notes: `bench/datasets/README.md`.

## Prepare FLEURS WER Dataset

Install optional dataset tooling:

```bash
python3 -m venv .venv-bench
. .venv-bench/bin/activate
pip install "datasets<4" soundfile faster-whisper
```

Prepare French FLEURS test data:

```bash
npm run bench:prepare-fleurs -- \
  --config fr_fr \
  --split test \
  --out bench/datasets/fleurs-fr-test.jsonl \
  --audio-dir bench/datasets/audio/fleurs-fr-test
```

Run WER/CER benchmark:

```bash
MANIFEST=bench/datasets/fleurs-fr-test.jsonl npm run bench:suite
```

For a quick pipeline check, add `--limit 10` to `bench:prepare-fleurs`.

## Prepare faster-whisper Speed Audio

Download the same `benchmark.m4a` used by upstream `faster-whisper`:

```bash
npm run bench:prepare-speed-audio
```

This creates:

- `bench/datasets/faster-whisper-speed/benchmark.m4a`
- `bench/datasets/faster-whisper-speed.jsonl`

Run speed benchmark:

```bash
MANIFEST=bench/datasets/faster-whisper-speed.jsonl npm run bench:suite
```

The manifest intentionally contains an empty reference transcript. Use its results for timing and RTF, not WER.

## Full Suite

After creating a manifest and installing the optional Python benchmark dependencies, run:

```bash
MANIFEST=bench/datasets/common-voice-fr-test.jsonl npm run bench:suite
```

Useful profile knobs:

```bash
MANIFEST=bench/datasets/common-voice-fr-test.jsonl \
VOXTRAL_DECODER=ffmpeg \
VOXTRAL_MODE=warm \
VOXTRAL_DEVICE=cpu \
VOXTRAL_DTYPE=q4 \
FASTER_WHISPER_MODEL=small \
FASTER_WHISPER_DEVICE=cpu \
FASTER_WHISPER_COMPUTE_TYPE=int8 \
npm run bench:suite
```

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
- Report FLEURS quality and `benchmark.m4a` speed as separate tables.

## Current Status

- Harness exists.
- FLEURS prep exists for WER/CER.
- Upstream faster-whisper speed audio prep exists for velocity.
- No benchmark result is claimed yet.
- The next step is to run both manifests on the same machine and commit the generated summaries separately if we want published numbers.
