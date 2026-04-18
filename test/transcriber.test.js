import test from "node:test";
import assert from "node:assert/strict";
import { VoxtralTranscriber, DEFAULT_MODEL } from "../dist/index.node.js";

function createFakeRuntime() {
  const calls = {
    disposed: 0,
    generate: 0,
    lastGenerateInput: undefined,
    lastLoadOptions: undefined,
    load: 0,
    processedAudioLengths: [],
  };

  const processor = Object.assign(
    async (audio) => {
      calls.processedAudioLengths.push(audio.length);
      return { input_features: audio };
    },
    {
      feature_extractor: {
        config: {
          sampling_rate: 16000,
        },
      },
      batch_decode() {
        return ["bonjour le monde"];
      },
    },
  );

  const model = {
    async dispose() {
      calls.disposed += 1;
    },
    async generate(input) {
      calls.generate += 1;
      calls.lastGenerateInput = input;
      return { fake: true };
    },
  };

  return {
    calls,
    runtime: {
      async load(options) {
        calls.load += 1;
        calls.lastLoadOptions = options;
        return { model, processor };
      },
    },
  };
}

function createFakeVoxtralRealtimeRuntime() {
  const calls = {
    generate: 0,
    lastGenerateInput: undefined,
    processorOptions: [],
    processedAudioLengths: [],
  };

  const processor = Object.assign(
    async (audio, options = {}) => {
      calls.processedAudioLengths.push(audio.length);
      calls.processorOptions.push(options);
      return {
        input_features: { samples: audio.length },
        input_ids: options.is_first_audio_chunk ? { bos: true } : undefined,
      };
    },
    {
      feature_extractor: {
        config: {
          sampling_rate: 16000,
        },
      },
      num_samples_first_audio_chunk: 4,
      num_samples_per_audio_chunk: 2,
      raw_audio_length_per_tok: 2,
      batch_decode() {
        return ["bonjour realtime"];
      },
    },
  );

  const model = {
    async dispose() {},
    async generate(input) {
      calls.generate += 1;
      calls.lastGenerateInput = input;
      return { fake: true };
    },
  };

  return {
    calls,
    runtime: {
      async load() {
        return { model, processor };
      },
    },
  };
}

function createFakeAudioDecoder() {
  const calls = {
    decodeFile: 0,
    lastOptions: undefined,
  };

  return {
    calls,
    decoder: {
      name: "fake-decoder",
      async decodeFile(_path, options) {
        calls.decodeFile += 1;
        calls.lastOptions = options;
        return {
          sampleRate: options?.sampleRate ?? 8000,
          samples: new Float32Array([0, 1, 0, -1]),
        };
      },
    },
  };
}

test("VoxtralTranscriber lazily loads runtime and transcribes raw audio", async () => {
  const fake = createFakeRuntime();
  const transcriber = new VoxtralTranscriber(
    {
      model: DEFAULT_MODEL,
    },
    fake.runtime,
  );

  const result = await transcriber.transcribeAudio(new Float32Array([0, 0.5, 0, -0.5]), {
    sampleRate: 16000,
  });

  assert.equal(fake.calls.load, 1);
  assert.equal(fake.calls.generate, 1);
  assert.equal(result.decoder, "internal-wav");
  assert.equal(result.text, "bonjour le monde");

  await transcriber.dispose();
  assert.equal(fake.calls.disposed, 1);
});

test("VoxtralTranscriber resamples raw audio before inference", async () => {
  const fake = createFakeRuntime();
  const transcriber = new VoxtralTranscriber({}, fake.runtime);

  await transcriber.transcribeAudio(new Float32Array([0, 1, 0, -1]), {
    sampleRate: 8000,
  });

  assert.equal(fake.calls.processedAudioLengths[0], 8);
});

test("VoxtralTranscriber chunks Voxtral Realtime audio before generation", async () => {
  const fake = createFakeVoxtralRealtimeRuntime();
  const transcriber = new VoxtralTranscriber({}, fake.runtime);

  const result = await transcriber.transcribeAudio(new Float32Array([0, 1, 2, 3, 4, 5, 6]), {
    sampleRate: 16000,
  });

  assert.equal(fake.calls.generate, 1);
  assert.deepEqual(fake.calls.processedAudioLengths, [4, 2, 2]);
  assert.deepEqual(fake.calls.processorOptions, [
    { is_first_audio_chunk: true, is_streaming: true },
    { is_first_audio_chunk: false, is_streaming: true },
    { is_first_audio_chunk: false, is_streaming: true },
  ]);
  assert.equal(fake.calls.lastGenerateInput.input_ids.bos, true);
  assert.equal(fake.calls.lastGenerateInput.input_features.length, 3);
  assert.equal(fake.calls.lastGenerateInput.max_new_tokens, 68);
  assert.equal(result.text, "bonjour realtime");
});

test("VoxtralTranscriber uses the configured audio decoder backend for file inputs", async () => {
  const fakeRuntime = createFakeRuntime();
  const fakeDecoder = createFakeAudioDecoder();
  const transcriber = new VoxtralTranscriber(
    {},
    fakeRuntime.runtime,
    fakeDecoder.decoder,
  );

  const result = await transcriber.transcribeFile("/tmp/sample.wav");

  assert.equal(fakeDecoder.calls.decodeFile, 1);
  assert.equal(fakeRuntime.calls.generate, 1);
  assert.equal(result.decoder, "fake-decoder");
  assert.equal(fakeDecoder.calls.lastOptions.sampleRate, 16000);
  assert.equal(fakeDecoder.calls.lastOptions.channels, 1);
  assert.equal(fakeRuntime.calls.processedAudioLengths[0], 4);
});

test("VoxtralTranscriber can require a pre-provisioned local model without runtime fetch", async () => {
  const fake = createFakeRuntime();
  const transcriber = new VoxtralTranscriber(
    {
      model: DEFAULT_MODEL,
      modelPath: "/opt/models/voxtral-mini-4b-realtime",
      requireLocalModel: true,
    },
    fake.runtime,
  );

  const result = await transcriber.transcribeAudio(new Float32Array([0, 0.5, 0, -0.5]), {
    sampleRate: 16000,
  });

  assert.equal(fake.calls.load, 1);
  assert.equal(fake.calls.lastLoadOptions.model, DEFAULT_MODEL);
  assert.equal(fake.calls.lastLoadOptions.modelPath, "/opt/models/voxtral-mini-4b-realtime");
  assert.equal(fake.calls.lastLoadOptions.requireLocalModel, true);
  assert.equal(fake.calls.lastLoadOptions.localFilesOnly, true);
  assert.equal(result.model, "/opt/models/voxtral-mini-4b-realtime");
});
