import { spawn, type SpawnOptionsWithoutStdio } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { Readable } from "node:stream";
import { Buffer } from "node:buffer";
import { decodeWav, readWavFile } from "./audio.node.js";

export interface DecodedAudio {
  channels?: number;
  sampleRate: number;
  samples: Float32Array;
}

export type VoxtralTarget = "auto" | "browser" | "node";
export type AudioDecoderInput = Blob | string | URL;

export interface DecodeFileOptions {
  channels?: number;
  sampleRate?: number;
}

export interface AudioDecoderBackend {
  readonly name: string;
  decodeFile(input: AudioDecoderInput, options?: DecodeFileOptions): Promise<DecodedAudio>;
}

type SpawnFactory = (
  command: string,
  args: readonly string[],
  options: SpawnOptionsWithoutStdio,
) => {
  stdout: Readable;
  stderr: Readable;
  once(event: "error", listener: (error: Error) => void): unknown;
  once(event: "close", listener: (code: number | null) => void): unknown;
};

type AudioContextLike = {
  close?: () => Promise<void>;
  decodeAudioData(audioData: ArrayBuffer): Promise<{
    length: number;
    numberOfChannels: number;
    sampleRate: number;
    getChannelData(channel: number): Float32Array;
  }>;
};

function isBlobLike(value: unknown): value is Blob {
  return typeof Blob !== "undefined" && value instanceof Blob;
}

function downmixToMono(channelData: Float32Array[]): Float32Array {
  if (channelData.length === 0) {
    return new Float32Array();
  }
  if (channelData.length === 1) {
    return channelData[0].slice();
  }

  const samples = new Float32Array(channelData[0].length);
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
    let sum = 0;
    for (const channel of channelData) {
      sum += channel[sampleIndex] ?? 0;
    }
    samples[sampleIndex] = sum / channelData.length;
  }
  return samples;
}

function concatenateUint8Arrays(chunks: Uint8Array[]): Uint8Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function float32FromBytes(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`Invalid ffmpeg PCM output size: expected a multiple of 4 bytes, got ${bytes.byteLength}.`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getFloat32(index * 4, true);
  }
  return samples;
}

async function normalizeFfmpegInput(input: AudioDecoderInput): Promise<string> {
  if (isBlobLike(input)) {
    throw new Error('FfmpegDecoder does not accept Blob/File inputs. Use BrowserNativeAudioDecoder or transcribeAudio().');
  }
  if (input instanceof URL && input.protocol === "file:") {
    return fileURLToPath(input);
  }
  return input.toString();
}

export class InternalWavDecoder implements AudioDecoderBackend {
  readonly name = "internal-wav";

  async decodeFile(input: AudioDecoderInput): Promise<DecodedAudio> {
    if (isBlobLike(input)) {
      return decodeWav(new Uint8Array(await input.arrayBuffer()));
    }
    return await readWavFile(input);
  }
}

export interface FfmpegDecoderOptions {
  channels?: number;
  ffmpegPath?: string;
  sampleRate?: number;
  spawnFactory?: SpawnFactory;
}

export class FfmpegDecoder implements AudioDecoderBackend {
  readonly name = "ffmpeg";

  private readonly channels: number;
  private readonly ffmpegPath: string;
  private readonly sampleRate: number;
  private readonly spawnFactory: SpawnFactory;

  constructor(options: FfmpegDecoderOptions = {}) {
    this.channels = options.channels ?? 1;
    this.ffmpegPath = options.ffmpegPath ?? "ffmpeg";
    this.sampleRate = options.sampleRate ?? 16_000;
    this.spawnFactory =
      options.spawnFactory ??
      ((command, args, spawnOptions) => spawn(command, args as string[], spawnOptions));
  }

  async decodeFile(input: AudioDecoderInput, options: DecodeFileOptions = {}): Promise<DecodedAudio> {
    const channels = options.channels ?? this.channels;
    const sampleRate = options.sampleRate ?? this.sampleRate;
    const normalizedInput = await normalizeFfmpegInput(input);
    const child = this.spawnFactory(
      this.ffmpegPath,
      [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        normalizedInput,
        "-ac",
        String(channels),
        "-ar",
        String(sampleRate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
      ],
      { stdio: ["ignore", "pipe", "pipe"] } as SpawnOptionsWithoutStdio,
    );

    return await new Promise<DecodedAudio>((resolve, reject) => {
      const stdoutChunks: Uint8Array[] = [];
      const stderrChunks: Uint8Array[] = [];

      child.stdout.on("data", (chunk) => {
        stdoutChunks.push(chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
      });

      child.stderr.on("data", (chunk) => {
        stderrChunks.push(chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
      });

      child.once("error", (error) => {
        reject(new Error(`Failed to start ffmpeg decoder with "${this.ffmpegPath}": ${error.message}`));
      });

      child.once("close", (code) => {
        if (code !== 0) {
          const details = new TextDecoder().decode(concatenateUint8Arrays(stderrChunks)).trim();
          reject(
            new Error(
              details
                ? `ffmpeg decoder failed (${code}): ${details}`
                : `ffmpeg decoder failed with exit code ${code}.`,
            ),
          );
          return;
        }

        resolve({
          channels,
          sampleRate,
          samples: float32FromBytes(concatenateUint8Arrays(stdoutChunks)),
        });
      });
    });
  }
}

export interface BrowserNativeAudioDecoderOptions {
  audioContextFactory?: (options?: { sampleRate?: number }) => AudioContextLike;
  fetcher?: typeof fetch;
}

export class BrowserNativeAudioDecoder implements AudioDecoderBackend {
  readonly name = "browser-native";

  private readonly audioContextFactory?: BrowserNativeAudioDecoderOptions["audioContextFactory"];
  private readonly fetcher?: typeof fetch;

  constructor(options: BrowserNativeAudioDecoderOptions = {}) {
    this.audioContextFactory = options.audioContextFactory;
    this.fetcher = options.fetcher;
  }

  async decodeFile(input: AudioDecoderInput, options: DecodeFileOptions = {}): Promise<DecodedAudio> {
    const channels = options.channels ?? 1;
    if (channels !== 1) {
      throw new Error(`BrowserNativeAudioDecoder currently supports only mono output. Received channels=${channels}.`);
    }

    const fetcher = this.fetcher ?? globalThis.fetch;
    const audioContext = this.createAudioContext(options.sampleRate);
    const arrayBuffer = await this.readInputAsArrayBuffer(input, fetcher);

    try {
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const channelData = Array.from({ length: decoded.numberOfChannels }, (_, channel) => decoded.getChannelData(channel));

      return {
        channels,
        sampleRate: decoded.sampleRate,
        samples: downmixToMono(channelData),
      };
    } finally {
      await audioContext.close?.();
    }
  }

  private createAudioContext(sampleRate?: number): AudioContextLike {
    if (this.audioContextFactory) {
      return this.audioContextFactory(sampleRate ? { sampleRate } : undefined);
    }

    const AudioContextCtor =
      globalThis.AudioContext ??
      (globalThis as typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

    if (typeof AudioContextCtor !== "function") {
      throw new Error("BrowserNativeAudioDecoder requires AudioContext or webkitAudioContext.");
    }

    return new AudioContextCtor(sampleRate ? { sampleRate } : undefined);
  }

  private async readInputAsArrayBuffer(input: AudioDecoderInput, fetcher: typeof fetch | undefined): Promise<ArrayBuffer> {
    if (isBlobLike(input)) {
      return await input.arrayBuffer();
    }
    if (typeof fetcher !== "function") {
      throw new Error("BrowserNativeAudioDecoder requires fetch to load URL inputs.");
    }

    const response = await fetcher(input.toString());
    if (!response.ok) {
      throw new Error(`BrowserNativeAudioDecoder failed to fetch "${input.toString()}": ${response.status} ${response.statusText}`);
    }
    return await response.arrayBuffer();
  }
}

export function createDefaultAudioDecoderBackend(target: VoxtralTarget = "auto"): AudioDecoderBackend {
  if (target === "browser") {
    return new BrowserNativeAudioDecoder();
  }
  return new InternalWavDecoder();
}
