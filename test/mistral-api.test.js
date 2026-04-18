import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  MistralVoxtralApiClient,
  MistralVoxtralApiTranscriber,
  transcribeFileWithMistral,
} from "../dist/index.node.js";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "content-type": "application/json",
    },
    ...init,
  });
}

test("MistralVoxtralApiClient posts file URLs as JSON", async () => {
  const calls = [];
  const client = new MistralVoxtralApiClient({
    apiKey: "test-key",
    fetcher: async (url, init) => {
      calls.push({ init, url });
      return jsonResponse({
        language: "fr",
        model: "voxtral-mini-2507",
        text: "bonjour le monde",
      });
    },
  });

  const result = await client.transcribeUrl("https://example.com/audio.mp3", {
    diarize: false,
    language: "fr",
  });

  assert.equal(result.decoder, "mistral-api");
  assert.equal(result.provider, "mistral");
  assert.equal(result.model, "voxtral-mini-2507");
  assert.equal(result.language, "fr");
  assert.equal(result.text, "bonjour le monde");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://api.mistral.ai/v1/audio/transcriptions");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers.Authorization, "Bearer test-key");
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    diarize: false,
    file_url: "https://example.com/audio.mp3",
    language: "fr",
    model: "voxtral-mini-2602",
  });
});

test("MistralVoxtralApiTranscriber uploads local files as multipart form data", async () => {
  const dir = await mkdtemp(join(tmpdir(), "voxtral-mistral-test-"));
  const filePath = join(dir, "sample.wav");
  await writeFile(filePath, new Uint8Array([82, 73, 70, 70]));

  const transcriber = new MistralVoxtralApiTranscriber({
    apiKey: "test-key",
    fetcher: async (_url, init) => {
      const body = init.body;
      assert.equal(body instanceof FormData, true);
      assert.equal(body.get("model"), "voxtral-mini-2602");
      assert.equal(body.get("language"), "fr");
      const file = body.get("file");
      assert.equal(file.name, "sample.wav");
      assert.equal(file.type, "audio/wav");
      return jsonResponse({
        model: "voxtral-mini-2507",
        text: "transcrit",
      });
    },
  });

  const result = await transcriber.transcribeFile(filePath, {
    language: "fr",
  });

  assert.equal(result.text, "transcrit");
});

test("transcribeFileWithMistral accepts injected api key and fetcher", async () => {
  const result = await transcribeFileWithMistral("https://example.com/audio.wav", {
    apiKey: "test-key",
    fetcher: async () => jsonResponse({ text: "ok" }),
  });

  assert.equal(result.text, "ok");
});

test("MistralVoxtralApiClient fails fast without api key", async () => {
  const client = new MistralVoxtralApiClient({
    fetcher: async () => {
      throw new Error("fetch should not be called");
    },
  });

  await assert.rejects(
    client.transcribeUrl("https://example.com/audio.mp3"),
    /requires apiKey or MISTRAL_API_KEY/,
  );
});
