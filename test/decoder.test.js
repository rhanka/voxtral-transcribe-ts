import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { Buffer } from "node:buffer";
import {
  BrowserNativeAudioDecoder,
  createDefaultAudioDecoderBackend,
  FfmpegDecoder,
  InternalWavDecoder,
} from "../dist/index.node.js";

function createMockFfmpegProcess({ exitCode = 0, stderr = "", stdout = Buffer.alloc(0) } = {}) {
  const child = new EventEmitter();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();

  queueMicrotask(() => {
    if (stdout.length > 0) {
      child.stdout.write(stdout);
    }
    child.stdout.end();

    if (stderr.length > 0) {
      child.stderr.write(stderr);
    }
    child.stderr.end();

    child.emit("close", exitCode);
  });

  return child;
}

test("FfmpegDecoder spawns ffmpeg and decodes float PCM output", async () => {
  const calls = [];
  const pcm = new Float32Array([0, 0.5, -0.5]);
  const stdout = Buffer.from(pcm.buffer.slice(0));
  const decoder = new FfmpegDecoder({
    ffmpegPath: "/usr/bin/ffmpeg",
    spawnFactory(command, args, options) {
      calls.push({ args, command, options });
      return createMockFfmpegProcess({ stdout });
    },
  });

  const result = await decoder.decodeFile("/tmp/input.mp3", {
    channels: 1,
    sampleRate: 16000,
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "/usr/bin/ffmpeg");
  assert.deepEqual(calls[0].args, [
    "-hide_banner",
    "-loglevel",
    "error",
    "-nostdin",
    "-i",
    "/tmp/input.mp3",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-f",
    "f32le",
    "-acodec",
    "pcm_f32le",
    "pipe:1",
  ]);
  assert.equal(result.sampleRate, 16000);
  assert.equal(result.channels, 1);
  assert.deepEqual(Array.from(result.samples), [0, 0.5, -0.5]);
});

test("BrowserNativeAudioDecoder decodes fetched audio and downmixes to mono", async () => {
  const calls = {
    close: 0,
    fetch: 0,
  };
  const decoder = new BrowserNativeAudioDecoder({
    audioContextFactory(options) {
      assert.equal(options?.sampleRate, 16000);
      return {
        async close() {
          calls.close += 1;
        },
        async decodeAudioData() {
          return {
            length: 3,
            numberOfChannels: 2,
            sampleRate: 16000,
            getChannelData(channel) {
              return channel === 0
                ? new Float32Array([1, 0, -1])
                : new Float32Array([0, 1, 0]);
            },
          };
        },
      };
    },
    async fetcher(url) {
      calls.fetch += 1;
      assert.equal(url, "https://example.com/audio.mp3");
      return {
        ok: true,
        async arrayBuffer() {
          return new ArrayBuffer(8);
        },
      };
    },
  });

  const result = await decoder.decodeFile("https://example.com/audio.mp3", {
    channels: 1,
    sampleRate: 16000,
  });

  assert.equal(calls.fetch, 1);
  assert.equal(calls.close, 1);
  assert.equal(result.channels, 1);
  assert.equal(result.sampleRate, 16000);
  assert.deepEqual(Array.from(result.samples), [0.5, 0.5, -0.5]);
});

test("createDefaultAudioDecoderBackend falls back to InternalWavDecoder in Node", () => {
  const decoder = createDefaultAudioDecoderBackend();

  assert.ok(decoder instanceof InternalWavDecoder);
});

test("createDefaultAudioDecoderBackend respects explicit target selection", () => {
  assert.ok(createDefaultAudioDecoderBackend("browser") instanceof BrowserNativeAudioDecoder);
  assert.ok(createDefaultAudioDecoderBackend("node") instanceof InternalWavDecoder);
});
