# Feature: Multi-target Voxtral Transcription

## Objective

Build `voxtral-transcribe-ts` as a lightweight package that supports the same plugin surface across:

- `client-side/browser`
- `local/server`

The package must keep inference and media decoding separated so each environment can use the most appropriate backend.

## Scope / Guardrails

- [x] Keep a `Node/TypeScript` baseline with `0 Python`.
- [x] Support both official targets:
  - [x] `client-side/browser`
  - [x] `local/server`
- [x] Keep `inference runtime` and `audio/media decoder` as separate layers.
- [x] Keep the public API aligned across environments.
- [x] Allow the server/local target to use the most performant decoder/runtime combination.
- [x] Keep a minimal fallback path that works without `ffmpeg`.

## Current Status

### Already implemented

- [x] npm package scaffold exists
- [x] local Node transcriber exists
- [x] internal WAV decoder exists
- [x] ONNX-based local runtime exists
- [x] build passes
- [x] tests pass
- [x] npm pack dry-run passes

### Not implemented yet

- [ ] codec support matrix in docs
- [ ] browser inference runtime qualification in a real browser app

## Decisions Already Taken

### Audio decoder strategy

- [x] `internal-wav` is kept as the minimal fallback decoder
- [x] `ffmpeg` is the default decoder for `local/server`
- [x] `browser-native` is the official decoder for `client-side/browser`

### Inference runtime strategy

- [x] Browser runtime target:
  - [x] `@huggingface/transformers`
  - [x] `onnxruntime-web`
  - [x] `WebGPU` first
  - [x] `WASM` fallback
- [x] Local/server runtime target:
  - [x] `@huggingface/transformers`
  - [x] `onnxruntime-node`
- [x] Server-side may later expose a more aggressive high-performance backend if product priorities shift from model fidelity to throughput/latency.

## Decision Matrix

### Official browser port

- [x] Inference: `transformers.js` + `onnxruntime-web`
- [x] Decoder: `browser-native`
- [x] Browser-safe package entry shipped
- [ ] Inference runtime qualified in a real browser app

### Official local Node port

- [x] Inference: `transformers.js` + `onnxruntime-node`
- [x] Decoder: `internal-wav` today
- [x] Decoder: `ffmpeg` added

### Official high-performance server path

- [x] Inference baseline: `onnxruntime-node`
- [x] Decoder baseline: `ffmpeg`
- [ ] Hardware-accelerated server profile documented
- [ ] Optional alternative server runtime explored

## Plan / Todo

- [x] **Lot 0 — Baseline**
  - [x] Create the package scaffold
  - [x] Publishable npm layout
  - [x] Add TypeScript build
  - [x] Add smoke tests
  - [x] Validate local Node baseline

- [x] **Lot 1 — Current local MVP**
  - [x] Add the current local transcriber
  - [x] Add the internal WAV decoder
  - [x] Keep a pure Node/TS path
  - [x] Validate build, tests, and pack

- [ ] **Lot 2 — Backend abstraction**
  - [x] Introduce `InferenceBackend`
  - [x] Introduce `AudioDecoderBackend`
  - [x] Make the current implementation the first backend pair
  - [x] Keep the public API stable while moving internals behind interfaces

- [ ] **Lot 3 — Local/server audio decoding**
  - [x] Add `FfmpegDecoder`
  - [x] Support multiformat input through `ffmpeg`
  - [x] Keep `InternalWavDecoder` as fallback
  - [x] Define backend selection rules for Node/local/server
  - [ ] Document operational requirements for `ffmpeg` and `ffmpeg-static`

- [ ] **Lot 4 — Browser audio decoding**
  - [x] Add `BrowserNativeAudioDecoder`
  - [x] Normalize browser file input to PCM
  - [ ] Document supported codecs as “browser-dependent”
  - [x] Add browser-specific tests or validation notes

- [ ] **Lot 5 — Browser inference port**
  - [x] Add browser-safe runtime wiring
  - [x] Split environment-specific entry points if needed
  - [ ] Define the default browser model strategy
  - [ ] Avoid making `Voxtral Mini 4B Realtime` the default browser model
  - [ ] Document expected constraints for WebGPU and WASM fallback

- [ ] **Lot 6 — Environment selection**
  - [x] Add explicit backend selection options
  - [x] Add automatic environment detection
  - [ ] Ensure the plugin works in both versions with the same high-level API
  - [ ] Add failure modes with actionable errors when a backend is unavailable

- [ ] **Lot 7 — Documentation consolidation**
  - [ ] Add a clear environment matrix:
    - [ ] browser
    - [ ] local Node
    - [ ] server
  - [ ] Add a clear decoder matrix:
    - [ ] `internal-wav`
    - [ ] `ffmpeg`
    - [ ] `browser-native`
  - [ ] Add examples for each environment
  - [ ] Add install guidance for lightweight vs full-featured usage

- [ ] **Lot 8 — Optional high-performance server path**
  - [ ] Reassess whether strict Voxtral compatibility remains the priority
  - [ ] Explore an alternative server backend only if product goals justify divergence
  - [ ] Compare candidates such as `whisper.cpp` or `sherpa-onnx`
  - [ ] Keep this optional and out of the baseline package path

## Where We Are Now

- [x] We have the local Node baseline
- [x] We have the minimal decoder baseline
- [x] We have the package foundation
- [x] We have backend abstraction for inference and decoding
- [x] We have a multiformat local/server decoder backend
- [x] We have the browser decoder
- [x] We have browser-safe packaging and entry points
- [ ] We do not yet have a browser-runtime qualification in a real app
- [x] We have automatic decoder selection by environment

## References

- https://github.com/huggingface/transformers.js
- https://huggingface.co/docs/transformers.js/en/guides/webgpu
- https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- https://www.npmjs.com/package/onnxruntime-node
- https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/decodeAudioData
- https://developer.mozilla.org/en-US/docs/Web/API/AudioDecoder
- https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Audio_codecs
- https://www.npmjs.com/package/ffmpeg-static
- https://ffmpegwasm.netlify.app/docs/performance/
- https://github.com/k2-fsa/sherpa-onnx
- https://github.com/ggml-org/whisper.cpp
- https://openbenchmarking.org/test/pts/whisper-cpp
- https://www.phoronix.com/news/Whisper-cpp-1.8.3-12x-Perf
