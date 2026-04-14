import test from "node:test";
import assert from "node:assert/strict";
import { decodeWav, resampleAudio } from "../dist/index.node.js";

function createStereo16BitWav() {
  const sampleRate = 8000;
  const channels = 2;
  const bitsPerSample = 16;
  const samplesPerChannel = [
    [0, 16384, -16384, 8192],
    [0, -16384, 16384, -8192],
  ];
  const blockAlign = channels * (bitsPerSample / 8);
  const dataSize = samplesPerChannel[0].length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeFourCC = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeFourCC(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeFourCC(8, "WAVE");
  writeFourCC(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeFourCC(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let frame = 0; frame < samplesPerChannel[0].length; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      view.setInt16(offset, samplesPerChannel[channel][frame], true);
      offset += 2;
    }
  }

  return new Uint8Array(buffer);
}

test("decodeWav decodes PCM WAV and mixes channels down to mono", () => {
  const decoded = decodeWav(createStereo16BitWav());

  assert.equal(decoded.channels, 2);
  assert.equal(decoded.sampleRate, 8000);
  assert.equal(decoded.samples.length, 4);
  assert.ok(Math.abs(decoded.samples[0] - 0) < 1e-6);
  assert.ok(Math.abs(decoded.samples[1] - 0) < 1e-6);
  assert.ok(Math.abs(decoded.samples[2] - 0) < 1e-6);
});

test("resampleAudio linearly resamples mono PCM", () => {
  const input = new Float32Array([0, 1, 0, -1]);
  const output = resampleAudio(input, 4000, 8000);

  assert.equal(output.length, 8);
  assert.ok(output[1] > 0 && output[1] < 1);
  assert.ok(output[5] < 0);
});
