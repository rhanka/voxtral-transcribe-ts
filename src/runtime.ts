import type { DataType, DeviceType } from "@huggingface/transformers";

export interface InferenceBackendLoadOptions {
  cacheDir?: string;
  device: DeviceType;
  dtype: DataType;
  localFilesOnly?: boolean;
  model: string;
  progressCallback?: (progress: unknown) => void;
  revision?: string;
}

export type ProcessorLike = {
  feature_extractor: {
    config: {
      sampling_rate: number;
    };
  };
  batch_decode(output: unknown, options?: { skip_special_tokens?: boolean }): string[];
} & ((audio: Float32Array) => Promise<Record<string, unknown>>);

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
    const { AutoProcessor, PreTrainedModel } = await import("@huggingface/transformers");
    const shared = {
      cache_dir: options.cacheDir,
      local_files_only: options.localFilesOnly,
      progress_callback: options.progressCallback,
      revision: options.revision,
    };

    const processor = (await AutoProcessor.from_pretrained(options.model, shared)) as ProcessorLike;
    const model = (await PreTrainedModel.from_pretrained(options.model, {
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
