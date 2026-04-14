import { decodeWav, readWavFile } from "./audio.browser.js";

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
}

export class FfmpegDecoder implements AudioDecoderBackend {
  readonly name = "ffmpeg";

  constructor(_options: FfmpegDecoderOptions = {}) {}

  async decodeFile(): Promise<DecodedAudio> {
    throw new Error("FfmpegDecoder is not available in browser builds.");
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
  if (target === "node") {
    return new InternalWavDecoder();
  }
  return new BrowserNativeAudioDecoder();
}
