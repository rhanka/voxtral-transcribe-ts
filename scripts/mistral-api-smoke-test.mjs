#!/usr/bin/env node

import { MistralVoxtralApiClient, DEFAULT_MISTRAL_TRANSCRIPTION_MODEL } from "../dist/index.node.js";

function createSilenceWav({ durationMs = 750, sampleRate = 16_000 } = {}) {
  const samples = Math.max(1, Math.floor((durationMs / 1000) * sampleRate));
  const dataBytes = samples * 2;
  const bytes = new Uint8Array(44 + dataBytes);
  const view = new DataView(bytes.buffer);

  writeAscii(bytes, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(bytes, 8, "WAVE");
  writeAscii(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(bytes, 36, "data");
  view.setUint32(40, dataBytes, true);

  return bytes;
}

function writeAscii(bytes, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    bytes[offset + index] = value.charCodeAt(index);
  }
}

if (!process.env.MISTRAL_API_KEY) {
  console.log("MISTRAL_API_KEY is not set; skipping Mistral API smoke test.");
  process.exit(0);
}

const client = new MistralVoxtralApiClient({
  apiKey: process.env.MISTRAL_API_KEY,
});
const audio = new Blob([createSilenceWav()], { type: "audio/wav" });
const result = await client.transcribeBlob(audio, {
  language: "fr",
}, "smoke.wav");

if (result.decoder !== "mistral-api") {
  throw new Error(`Unexpected decoder: ${result.decoder}`);
}

if (result.model !== DEFAULT_MISTRAL_TRANSCRIPTION_MODEL) {
  throw new Error(`Unexpected model: ${result.model}`);
}

console.log(`Mistral API smoke test passed with ${result.model}.`);
