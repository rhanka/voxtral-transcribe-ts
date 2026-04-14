import test from "node:test";
import assert from "node:assert/strict";
import { VoxtralTranscriber, DEFAULT_MODEL } from "../dist/index.node.js";

function createFakeRuntime() {
  const calls = {
    disposed: 0,
    generate: 0,
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
    async generate() {
      calls.generate += 1;
      return { fake: true };
    },
  };

  return {
    calls,
    runtime: {
      async load() {
        calls.load += 1;
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
