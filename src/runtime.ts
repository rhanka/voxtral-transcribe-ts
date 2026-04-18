import type { DataType, DeviceType } from "@huggingface/transformers";

export interface InferenceBackendLoadOptions {
  cacheDir?: string;
  device: DeviceType;
  dtype: DataType;
  localFilesOnly?: boolean;
  model: string;
  modelPath?: string;
  progressCallback?: (progress: unknown) => void;
  requireLocalModel?: boolean;
  revision?: string;
}

export type ProcessorLike = {
  feature_extractor: {
    config: {
      sampling_rate: number;
    };
  };
  num_samples_first_audio_chunk?: number;
  num_samples_per_audio_chunk?: number;
  raw_audio_length_per_tok?: number;
  batch_decode(output: unknown, options?: { skip_special_tokens?: boolean }): string[];
} & ((audio: Float32Array, options?: Record<string, unknown>) => Promise<Record<string, unknown>>);

export interface ModelLike {
  dispose(): Promise<unknown>;
  generate(input: Record<string, unknown>): Promise<unknown>;
}

export interface InferenceBackend {
  load(options: InferenceBackendLoadOptions): Promise<{
    model: ModelLike;
    processor: ProcessorLike;
  }>;
}

export class TransformersInferenceBackend implements InferenceBackend {
  async load(options: InferenceBackendLoadOptions): Promise<{ model: ModelLike; processor: ProcessorLike }> {
    const { AutoProcessor, PreTrainedModel, VoxtralRealtimeForConditionalGeneration } = await import("@huggingface/transformers");
    const modelSource = options.modelPath ?? options.model;
    const localFilesOnly = options.requireLocalModel || options.localFilesOnly || Boolean(options.modelPath);
    const shared = {
      cache_dir: options.cacheDir,
      local_files_only: localFilesOnly,
      progress_callback: options.progressCallback,
      revision: options.revision,
    };

    const processor = (await AutoProcessor.from_pretrained(modelSource, shared)) as ProcessorLike;
    const ModelClass = modelSource.toLowerCase().includes("voxtral")
      ? VoxtralRealtimeForConditionalGeneration
      : PreTrainedModel;
    const model = (await ModelClass.from_pretrained(modelSource, {
      ...shared,
      device: options.device,
      dtype: options.dtype,
    })) as ModelLike;

    return { model, processor };
  }
}

export type RuntimeLoadOptions = InferenceBackendLoadOptions;
export type VoxtralRuntime = InferenceBackend;
export class TransformersRuntime extends TransformersInferenceBackend {}
