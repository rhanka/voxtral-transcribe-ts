export const DEFAULT_MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1";
export const DEFAULT_MISTRAL_TRANSCRIPTION_MODEL = "voxtral-mini-latest";

export type MistralTimestampGranularity = "segment" | "word";

export interface MistralVoxtralApiClientOptions {
  apiKey?: string;
  baseUrl?: string;
  fetcher?: typeof fetch;
  model?: string;
}

export interface MistralVoxtralApiTranscribeOptions {
  contextBias?: string[];
  diarize?: boolean;
  language?: string;
  temperature?: number;
  timestampGranularities?: MistralTimestampGranularity[];
}

export interface MistralVoxtralApiTranscriptionResult {
  decoder: "mistral-api";
  durationMs: number;
  language?: string | null;
  model: string;
  provider: "mistral";
  segments?: unknown[];
  text: string;
  usage?: unknown;
}

type MistralTranscriptionResponse = {
  language?: string | null;
  model?: string;
  segments?: unknown[];
  text?: string;
  usage?: unknown;
};

function appendMultipartField(form: FormData, key: string, value: unknown): void {
  if (value === undefined || value === null) {
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      form.append(key, String(item));
    }
    return;
  }
  form.append(key, String(value));
}

function createRequestPayload(
  model: string,
  options: MistralVoxtralApiTranscribeOptions,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...extra,
    context_bias: options.contextBias,
    diarize: options.diarize,
    language: options.language,
    model,
    temperature: options.temperature,
    timestamp_granularities: options.timestampGranularities,
  };
}

function compactJsonPayload(payload: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(payload).filter(([, value]) => value !== undefined && value !== null));
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

async function readErrorBody(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  return text.trim();
}

export class MistralVoxtralApiClient {
  private readonly apiKey?: string;
  private readonly baseUrl: string;
  private readonly fetcher?: typeof fetch;
  private readonly model: string;

  constructor(options: MistralVoxtralApiClientOptions = {}) {
    this.apiKey = options.apiKey;
    this.baseUrl = normalizeBaseUrl(options.baseUrl ?? DEFAULT_MISTRAL_API_BASE_URL);
    this.fetcher = options.fetcher;
    this.model = options.model ?? DEFAULT_MISTRAL_TRANSCRIPTION_MODEL;
  }

  async transcribeBlob(
    audio: Blob,
    options: MistralVoxtralApiTranscribeOptions = {},
    fileName = "audio.wav",
  ): Promise<MistralVoxtralApiTranscriptionResult> {
    const form = new FormData();
    form.append("model", this.model);
    form.append("file", audio, fileName);
    appendMultipartField(form, "context_bias", options.contextBias);
    appendMultipartField(form, "diarize", options.diarize);
    appendMultipartField(form, "language", options.language);
    appendMultipartField(form, "temperature", options.temperature);
    appendMultipartField(form, "timestamp_granularities", options.timestampGranularities);

    return await this.sendTranscriptionRequest(form);
  }

  async transcribeUrl(
    fileUrl: string | URL,
    options: MistralVoxtralApiTranscribeOptions = {},
  ): Promise<MistralVoxtralApiTranscriptionResult> {
    const payload = compactJsonPayload(createRequestPayload(this.model, options, {
      file_url: fileUrl.toString(),
    }));

    return await this.sendTranscriptionRequest(JSON.stringify(payload), {
      "Content-Type": "application/json",
    });
  }

  private async sendTranscriptionRequest(
    body: BodyInit,
    extraHeaders: Record<string, string> = {},
  ): Promise<MistralVoxtralApiTranscriptionResult> {
    const apiKey = this.apiKey;
    if (!apiKey) {
      throw new Error("Mistral API transcription requires apiKey or MISTRAL_API_KEY.");
    }

    const fetcher = this.fetcher ?? globalThis.fetch;
    if (typeof fetcher !== "function") {
      throw new Error("Mistral API transcription requires fetch.");
    }

    const startedAt = performance.now();
    const response = await fetcher(`${this.baseUrl}/audio/transcriptions`, {
      body,
      headers: {
        ...extraHeaders,
        Authorization: `Bearer ${apiKey}`,
      },
      method: "POST",
    });
    const durationMs = performance.now() - startedAt;

    if (!response.ok) {
      const details = await readErrorBody(response);
      throw new Error(
        details
          ? `Mistral API transcription failed (${response.status} ${response.statusText}): ${details}`
          : `Mistral API transcription failed (${response.status} ${response.statusText}).`,
      );
    }

    const payload = await response.json() as MistralTranscriptionResponse;
    return {
      decoder: "mistral-api",
      durationMs,
      language: payload.language,
      model: payload.model ?? this.model,
      provider: "mistral",
      segments: payload.segments,
      text: payload.text ?? "",
      usage: payload.usage,
    };
  }
}
