# Benchmark Datasets

Keep benchmark audio outside git. This directory only stores manifest templates and notes.

## Manifest Contract

Every benchmark runner consumes JSONL:

```json
{"id":"fr_001","audio":"./audio/fr_001.wav","text":"bonjour je veux tester la transcription","language":"fr","durationMs":4200}
```

Paths are resolved relative to the manifest file, so a local layout like this is recommended:

```text
bench/datasets/
  manifest.fr.jsonl
  audio/
    fr_001.wav
    fr_002.wav
```

`bench/datasets/audio/`, `bench/datasets/raw/`, and generated manifests are gitignored.

## Common Voice

For a local Common Voice extract, use `validated.tsv` or `test.tsv` and the `clips/` directory:

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

This keeps the source audio as `mp3`. Use `FfmpegDecoder` for Voxtral:

```bash
VOXTRAL_DECODER=ffmpeg MANIFEST=bench/datasets/common-voice-fr-test.jsonl npm run bench:suite
```

## Internal Corpus

For an internal CSV/TSV export, include at least:

- stable sample id
- audio path or filename
- reference transcript
- optional language
- optional duration in milliseconds

Example:

```bash
npm run bench:manifest -- \
  --transcripts ./my-corpus/metadata.csv \
  --audio-dir ./my-corpus/audio \
  --out bench/datasets/internal-fr.jsonl \
  --id-column id \
  --audio-column file \
  --text-column transcript \
  --language fr
```
