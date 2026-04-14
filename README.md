# voxtral-transcribe-ts

Minimal TypeScript wrapper for local transcription with `Voxtral Mini 4B Realtime` in Node.js.

This package targets the ONNX checkpoint:

- `onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX`

It is intentionally small:

- Node/TS only, no Python
- thin wrapper around `@huggingface/transformers` + ONNX Runtime
- 0 external audio decoder dependency

The built-in file loader only supports `.wav` input so the package can stay lightweight. If you already have PCM samples in memory, use `transcribeAudio()`.

Architecture and multi-target rollout plan: [PLAN.md](/home/antoinefa/src/voxtral-ts/PLAN.md)

## Install

```bash
npm install voxtral-transcribe-ts
```

## Quick Start

```ts
import { VoxtralTranscriber } from "voxtral-transcribe-ts";

const transcriber = new VoxtralTranscriber({
  device: "cpu",
  dtype: "q4",
});

const result = await transcriber.transcribeFile("./sample.wav");
console.log(result.text);

await transcriber.dispose();
```

By default, the package now auto-selects the audio decoder backend:

- Node/local: `InternalWavDecoder`
- Browser: `BrowserNativeAudioDecoder`

The package now ships conditional entries:

- package root in Node -> `dist/index.node.js`
- package root in browser-aware bundlers -> `dist/index.browser.js`
- explicit subpaths:
  - `voxtral-transcribe-ts/node`
  - `voxtral-transcribe-ts/browser`

## Environment Matrix

| Environment | Package entry | Inference runtime | Default decoder | File input strategy |
|---|---|---|---|---|
| Node / local | `voxtral-transcribe-ts` or `voxtral-transcribe-ts/node` | `@huggingface/transformers` + `onnxruntime-node` | `InternalWavDecoder` | `wav` by default, multiformat via `FfmpegDecoder` |
| Browser | `voxtral-transcribe-ts` in browser-aware bundlers or `voxtral-transcribe-ts/browser` | browser-safe package entry | `BrowserNativeAudioDecoder` | URL, `Blob`, `File`, browser codec support dependent on runtime |
| Server high-perf | `voxtral-transcribe-ts/node` | `@huggingface/transformers` + `onnxruntime-node` | `FfmpegDecoder` recommended | multiformat through `ffmpeg` |

## Decoder Matrix

| Decoder | Environment | Purpose | Notes |
|---|---|---|---|
| `InternalWavDecoder` | Node, browser | Minimal fallback | `wav` only |
| `FfmpegDecoder` | Node / server | Best multiformat local path | Not available in browser builds |
| `BrowserNativeAudioDecoder` | Browser | Native client-side decoding | Depends on browser codec support |

You can override this with:

- `target: "auto" | "node" | "browser"`
- `audioDecoderBackend`
- `inferenceBackend`

## Raw Audio

```ts
import { transcribeAudio } from "voxtral-transcribe-ts";

const samples = new Float32Array([/* mono PCM samples */]);

const result = await transcribeAudio(samples, {
  sampleRate: 16_000,
});

console.log(result.text);
```

## API

### `new VoxtralTranscriber(options?)`

Options:

- `model`: defaults to `onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX`
- `device`: defaults to `cpu`
- `dtype`: defaults to `q4`
- `cacheDir`
- `localFilesOnly`
- `revision`
- `progressCallback`
- `target`: defaults to `auto`

### `await transcriber.load()`

Preloads the processor and model.

### `await transcriber.transcribeFile(path, options?)`

Reads a WAV file, downmixes it to mono, resamples it to the model sample rate, and returns:

```ts
type VoxtralTranscriptionResult = {
  decoder: string;
  durationMs: number;
  model: string;
  sampleRate: number;
  text: string;
};
```

### `await transcriber.transcribeAudio(samples, options?)`

Transcribes mono PCM samples already loaded in memory.

Options:

- `sampleRate`: defaults to `16000`
- `maxNewTokens`
- `skipSpecialTokens`: defaults to `true`

## Advanced

The transcriber now separates:

- inference backend
- audio decoder backend

The current default pair is:

- `TransformersInferenceBackend`
- `InternalWavDecoder` in Node
- `BrowserNativeAudioDecoder` in browsers

For multiformat local/server decoding, use `FfmpegDecoder`.

```ts
import { FfmpegDecoder, VoxtralTranscriber } from "voxtral-transcribe-ts";

const transcriber = new VoxtralTranscriber({
  audioDecoderBackend: new FfmpegDecoder(),
});

const result = await transcriber.transcribeFile("./sample.mp3");
console.log(result.text);
```

Browser inputs can be passed as URLs or `Blob` / `File` objects when using `BrowserNativeAudioDecoder` or the default browser auto-selection.

You can also create an instance through `createTranscriber(options)`, which uses the same defaults and target rules as `new VoxtralTranscriber(options)`.

## Browser Entry

```ts
import { createTranscriber } from "voxtral-transcribe-ts/browser";

const transcriber = createTranscriber({
  target: "browser",
});
```

## Node Entry

```ts
import { createTranscriber, FfmpegDecoder } from "voxtral-transcribe-ts/node";

const transcriber = createTranscriber({
  target: "node",
  audioDecoderBackend: new FfmpegDecoder(),
});
```

## WAV Support

The internal WAV decoder supports:

- PCM 8/16/24/32-bit
- IEEE float 32-bit
- mono or multi-channel input, mixed down to mono

For `mp3`, `m4a`, `ogg`, or `flac`, decode audio yourself and call `transcribeAudio()`.

If you want the package to decode those formats for you on local/server, instantiate the transcriber with `FfmpegDecoder`.

## Validation

```bash
npm run validate
npm run test:smoke
```

## CI / Release

The repository now ships a GitHub Actions workflow in `.github/workflows/typescript-ci.yml` modeled after `graphify`.

It does four things:

- runs `npm run validate` on Node `20` and `22`
- builds a tarball and installs it in a clean directory
- verifies the published root, `node`, and `browser` exports
- publishes to npm on tags matching `v*`

Publish strategy:

- default: GitHub Actions trusted publishing with `id-token: write`
- fallback: if `NPM_TOKEN` is configured as a repository secret, the workflow uses that token instead

Local pre-publish check:

```bash
npm run test:smoke
```

Typical release flow:

```bash
npm version patch
git push
git push --tags
```
