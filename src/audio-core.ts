export interface DecodedWav {
  channels: number;
  sampleRate: number;
  samples: Float32Array;
}

const RIFF_HEADER = "RIFF";
const WAVE_HEADER = "WAVE";
const FMT_CHUNK = "fmt ";
const DATA_CHUNK = "data";
const WAVE_FORMAT_PCM = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT = 0x0003;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;

function readFourCC(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function clampChunkSize(buffer: Uint8Array, offset: number, size: number): number {
  return Math.max(0, Math.min(size, buffer.byteLength - offset));
}

function decodeSample(view: DataView, offset: number, audioFormat: number, bitsPerSample: number): number {
  if (audioFormat === WAVE_FORMAT_IEEE_FLOAT) {
    if (bitsPerSample !== 32) {
      throw new Error(`Unsupported float WAV bit depth: ${bitsPerSample}.`);
    }
    return view.getFloat32(offset, true);
  }

  switch (bitsPerSample) {
    case 8:
      return (view.getUint8(offset) - 128) / 128;
    case 16:
      return view.getInt16(offset, true) / 32768;
    case 24: {
      const value =
        view.getUint8(offset) |
        (view.getUint8(offset + 1) << 8) |
        (view.getUint8(offset + 2) << 16);
      const signed = value & 0x800000 ? value | ~0xffffff : value;
      return signed / 8388608;
    }
    case 32:
      return view.getInt32(offset, true) / 2147483648;
    default:
      throw new Error(`Unsupported PCM WAV bit depth: ${bitsPerSample}.`);
  }
}

export function decodeWav(buffer: Uint8Array | ArrayBuffer): DecodedWav {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.byteLength < 44) {
    throw new Error("Invalid WAV file: file is too small.");
  }
  if (readFourCC(view, 0) !== RIFF_HEADER || readFourCC(view, 8) !== WAVE_HEADER) {
    throw new Error("Invalid WAV file: missing RIFF/WAVE header.");
  }

  let offset = 12;
  let audioFormat = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let blockAlign = 0;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= bytes.byteLength) {
    const chunkId = readFourCC(view, offset);
    const chunkSize = clampChunkSize(bytes, offset + 8, view.getUint32(offset + 4, true));
    const chunkDataOffset = offset + 8;

    if (chunkId === FMT_CHUNK) {
      if (chunkSize < 16) {
        throw new Error("Invalid WAV file: incomplete fmt chunk.");
      }
      let format = view.getUint16(chunkDataOffset, true);
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      blockAlign = view.getUint16(chunkDataOffset + 12, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);

      if (format === WAVE_FORMAT_EXTENSIBLE) {
        if (chunkSize < 40) {
          throw new Error("Unsupported WAV extensible format: fmt chunk is too small.");
        }
        const subFormat = view.getUint16(chunkDataOffset + 24, true);
        format = subFormat;
      }

      audioFormat = format;
    } else if (chunkId === DATA_CHUNK) {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (!audioFormat || !channels || !sampleRate || !bitsPerSample || !blockAlign) {
    throw new Error("Invalid WAV file: missing audio format metadata.");
  }
  if (dataOffset < 0 || dataSize <= 0) {
    throw new Error("Invalid WAV file: missing data chunk.");
  }
  if (audioFormat !== WAVE_FORMAT_PCM && audioFormat !== WAVE_FORMAT_IEEE_FLOAT) {
    throw new Error(`Unsupported WAV format: ${audioFormat}.`);
  }

  const frameCount = Math.floor(dataSize / blockAlign);
  const samples = new Float32Array(frameCount);
  const bytesPerSample = bitsPerSample / 8;

  for (let frame = 0; frame < frameCount; frame += 1) {
    let mono = 0;
    const frameOffset = dataOffset + frame * blockAlign;
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleOffset = frameOffset + channel * bytesPerSample;
      mono += decodeSample(view, sampleOffset, audioFormat, bitsPerSample);
    }
    samples[frame] = mono / channels;
  }

  return { channels, sampleRate, samples };
}

export function resampleAudio(
  samples: Float32Array | readonly number[],
  fromSampleRate: number,
  toSampleRate: number,
): Float32Array {
  if (fromSampleRate <= 0 || toSampleRate <= 0) {
    throw new Error("Sample rates must be strictly positive.");
  }

  const input = samples instanceof Float32Array ? samples : Float32Array.from(samples);
  if (input.length === 0) {
    return new Float32Array();
  }
  if (fromSampleRate === toSampleRate) {
    return input.slice();
  }

  const targetLength = Math.max(1, Math.round((input.length * toSampleRate) / fromSampleRate));
  const output = new Float32Array(targetLength);
  const ratio = fromSampleRate / toSampleRate;

  for (let index = 0; index < targetLength; index += 1) {
    const position = index * ratio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(leftIndex + 1, input.length - 1);
    const interpolation = position - leftIndex;
    output[index] = input[leftIndex] * (1 - interpolation) + input[rightIndex] * interpolation;
  }

  return output;
}
