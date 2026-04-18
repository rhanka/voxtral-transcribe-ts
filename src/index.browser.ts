import type { DataType, DeviceType } from "@huggingface/transformers";
import { resampleAudio, type DecodedWav } from "./audio.browser.js";
import {
  BrowserNativeAudioDecoder,
  createDefaultAudioDecoderBackend,
  FfmpegDecoder,
  InternalWavDecoder,
  type AudioDecoderBackend,
  type AudioDecoderInput,
  type VoxtralTarget,
} from "./decoder.browser.js";
import { TransformersInferenceBackend, type InferenceBackend, type ModelLike, type ProcessorLike } from "./runtime.js";

export type VoxtralDevice = DeviceType;
export type VoxtralDtype = DataType;

export const DEFAULT_MODEL = "onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX";
export const DEFAULT_DEVICE: VoxtralDevice = "cpu";
export const DEFAULT_DTYPE: VoxtralDtype = "q4";
export const DEFAULT_SAMPLE_RATE = 16_000;
export const DEFAULT_TARGET: VoxtralTarget = "auto";

export interface VoxtralTranscriberOptions {
  audioDecoderBackend?: AudioDecoderBackend;
  cacheDir?: string;
  device?: VoxtralDevice;
  dtype?: VoxtralDtype;
  inferenceBackend?: InferenceBackend;
  localFilesOnly?: boolean;
  model?: string;
  modelPath?: string;
  progressCallback?: (progress: unknown) => void;
  requireLocalModel?: boolean;
  revision?: string;
  target?: VoxtralTarget;
}

export interface VoxtralTranscribeOptions {
  maxNewTokens?: number;
  sampleRate?: number;
  skipSpecialTokens?: boolean;
}

export interface VoxtralTranscriptionResult {
  decoder: string;
  durationMs: number;
  model: string;
  sampleRate: number;
  text: string;
}

interface LoadedRuntime {
  model: ModelLike;
  processor: ProcessorLike;
}

interface NormalizedTranscriberOptions {
  cacheDir?: string;
  device: VoxtralDevice;
  dtype: VoxtralDtype;
  localFilesOnly: boolean;
  model: string;
  modelPath?: string;
  progressCallback?: (progress: unknown) => void;
  requireLocalModel: boolean;
  revision?: string;
}

function isVoxtralRealtimeProcessor(processor: ProcessorLike): processor is ProcessorLike & {
  num_samples_first_audio_chunk: number;
  num_samples_per_audio_chunk: number;
} {
  return Number.isFinite(processor.num_samples_first_audio_chunk)
    && Number.isFinite(processor.num_samples_per_audio_chunk)
    && Number(processor.num_samples_first_audio_chunk) > 0
    && Number(processor.num_samples_per_audio_chunk) > 0;
}

function resolveMaxNewTokens(
  processor: ProcessorLike,
  audio: Float32Array,
  options: VoxtralTranscribeOptions,
): number | undefined {
  if (options.maxNewTokens !== undefined) {
    return options.maxNewTokens;
  }

  if (!isVoxtralRealtimeProcessor(processor)) {
    return undefined;
  }

  const samplesPerTextStep = Number(processor.raw_audio_length_per_tok)
    || Math.max(1, Math.floor(processor.num_samples_per_audio_chunk / 8));
  return Math.ceil(audio.length / samplesPerTextStep) + 64;
}

async function createGenerationInputs(processor: ProcessorLike, audio: Float32Array): Promise<Record<string, unknown>> {
  if (!isVoxtralRealtimeProcessor(processor)) {
    return await processor(audio);
  }

  const inputFeatures: unknown[] = [];
  let inputIds: unknown;
  let offset = 0;
  let isFirstChunk = true;

  // Voxtral Realtime ONNX consumes fixed-size streaming feature chunks.
  // Padding the last chunk keeps the audio encoder projector shape valid.
  while (offset < audio.length || isFirstChunk) {
    const chunkSamples = isFirstChunk
      ? processor.num_samples_first_audio_chunk
      : processor.num_samples_per_audio_chunk;
    const chunk = new Float32Array(chunkSamples);
    chunk.set(audio.subarray(offset, Math.min(audio.length, offset + chunkSamples)));
    const chunkInputs = await processor(chunk, {
      is_first_audio_chunk: isFirstChunk,
      is_streaming: true,
    });

    if (isFirstChunk) {
      inputIds = chunkInputs.input_ids;
    }
    if (!chunkInputs.input_features) {
      throw new Error("Voxtral Realtime processor did not return input_features.");
    }

    inputFeatures.push(chunkInputs.input_features);
    offset += chunkSamples;
    isFirstChunk = false;
  }

  if (!inputIds) {
    throw new Error("Voxtral Realtime processor did not return input_ids for the first chunk.");
  }

  return {
    input_features: inputFeatures,
    input_ids: inputIds,
  };
}

export { decodeWav, readWavFile, resampleAudio, type DecodedWav } from "./audio.browser.js";
export {
  BrowserNativeAudioDecoder,
  createDefaultAudioDecoderBackend,
  FfmpegDecoder,
  InternalWavDecoder,
  type AudioDecoderBackend,
  type DecodedAudio,
  type DecodeFileOptions,
  type FfmpegDecoderOptions,
  type AudioDecoderInput,
  type BrowserNativeAudioDecoderOptions,
  type VoxtralTarget,
} from "./decoder.browser.js";
export {
  TransformersInferenceBackend,
  TransformersRuntime,
  type InferenceBackend,
  type InferenceBackendLoadOptions,
  type ModelLike,
  type ProcessorLike,
  type VoxtralRuntime,
} from "./runtime.js";

export function createDefaultInferenceBackend(): InferenceBackend {
  return new TransformersInferenceBackend();
}

export class VoxtralTranscriber {
  private readonly audioDecoderBackend: AudioDecoderBackend;
  private readonly inferenceBackend: InferenceBackend;
  private readonly options: NormalizedTranscriberOptions;
  private runtimePromise?: Promise<LoadedRuntime>;

  constructor(
    options: VoxtralTranscriberOptions = {},
    inferenceBackend: InferenceBackend = options.inferenceBackend ?? createDefaultInferenceBackend(),
    audioDecoderBackend: AudioDecoderBackend = options.audioDecoderBackend ?? createDefaultAudioDecoderBackend(options.target ?? DEFAULT_TARGET),
  ) {
    const modelPath = options.modelPath;
    const requireLocalModel = options.requireLocalModel ?? Boolean(modelPath);

    this.options = {
      cacheDir: options.cacheDir,
      device: options.device ?? DEFAULT_DEVICE,
      dtype: options.dtype ?? DEFAULT_DTYPE,
      localFilesOnly: options.localFilesOnly ?? requireLocalModel,
      model: options.model ?? DEFAULT_MODEL,
      modelPath,
      progressCallback: options.progressCallback,
      requireLocalModel,
      revision: options.revision,
    };
    this.inferenceBackend = inferenceBackend;
    this.audioDecoderBackend = audioDecoderBackend;
  }

  async load(): Promise<void> {
    await this.ensureRuntime();
  }

  async transcribeAudio(
    audio: Float32Array | readonly number[],
    options: VoxtralTranscribeOptions = {},
  ): Promise<VoxtralTranscriptionResult> {
    const { model, processor } = await this.ensureRuntime();
    const sourceSampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
    const targetSampleRate = processor.feature_extractor.config.sampling_rate ?? DEFAULT_SAMPLE_RATE;
    const preparedAudio = resampleAudio(audio, sourceSampleRate, targetSampleRate);

    const startedAt = performance.now();
    const inputs = await createGenerationInputs(processor, preparedAudio);
    const outputs = await model.generate({
      ...inputs,
      max_new_tokens: resolveMaxNewTokens(processor, preparedAudio, options),
    });
    const text = processor.batch_decode(outputs, { skip_special_tokens: options.skipSpecialTokens ?? true })[0]?.trim() ?? "";

    return {
      decoder: this.audioDecoderBackend.name,
      durationMs: performance.now() - startedAt,
      model: this.options.modelPath ?? this.options.model,
      sampleRate: targetSampleRate,
      text,
    };
  }

  async transcribeFile(
    path: AudioDecoderInput,
    options: Omit<VoxtralTranscribeOptions, "sampleRate"> = {},
  ): Promise<VoxtralTranscriptionResult> {
    const { processor } = await this.ensureRuntime();
    const targetSampleRate = processor.feature_extractor.config.sampling_rate ?? DEFAULT_SAMPLE_RATE;
    const decoded = await this.audioDecoderBackend.decodeFile(path, {
      channels: 1,
      sampleRate: targetSampleRate,
    });
    return await this.transcribeAudio(decoded.samples, {
      ...options,
      sampleRate: decoded.sampleRate,
    });
  }

  async dispose(): Promise<void> {
    const runtime = this.runtimePromise ? await this.runtimePromise : undefined;
    if (!runtime) {
      return;
    }
    await runtime.model.dispose();
    this.runtimePromise = undefined;
  }

  private async ensureRuntime(): Promise<LoadedRuntime> {
    if (!this.runtimePromise) {
      this.runtimePromise = this.inferenceBackend.load({
        cacheDir: this.options.cacheDir,
        device: this.options.device,
        dtype: this.options.dtype,
        localFilesOnly: this.options.localFilesOnly,
        model: this.options.model,
        modelPath: this.options.modelPath,
        progressCallback: this.options.progressCallback,
        requireLocalModel: this.options.requireLocalModel,
        revision: this.options.revision,
      });
    }
    return await this.runtimePromise;
  }
}

export function createTranscriber(options: VoxtralTranscriberOptions = {}): VoxtralTranscriber {
  return new VoxtralTranscriber(options);
}

export async function transcribeAudio(
  audio: Float32Array | readonly number[],
  options: VoxtralTranscriberOptions & VoxtralTranscribeOptions = {},
): Promise<VoxtralTranscriptionResult> {
  const transcriber = new VoxtralTranscriber(options);
  try {
    return await transcriber.transcribeAudio(audio, options);
  } finally {
    await transcriber.dispose();
  }
}

export async function transcribeFile(
  path: AudioDecoderInput,
  options: VoxtralTranscriberOptions & Omit<VoxtralTranscribeOptions, "sampleRate"> = {},
): Promise<VoxtralTranscriptionResult> {
  const transcriber = new VoxtralTranscriber(options);
  try {
    return await transcriber.transcribeFile(path, options);
  } finally {
    await transcriber.dispose();
  }
}
