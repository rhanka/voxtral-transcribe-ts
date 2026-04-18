#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, createWriteStream } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PACKAGE_JSON = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf8"));

function parseArgs(argv) {
  const args = {
    decoder: "internal-wav",
    device: "cpu",
    dtype: "q4",
    mode: "warm",
    out: "bench/results/voxtral.jsonl",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }

    const key = arg.slice(2);
    if (key === "local-files-only" || key === "require-local-model") {
      args[toCamelCase(key)] = true;
      continue;
    }

    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    args[toCamelCase(key)] = value;
    i += 1;
  }

  if (!args.manifest) {
    throw new Error("Missing --manifest");
  }

  if (args.mode !== "warm" && args.mode !== "cold") {
    throw new Error("--mode must be warm or cold");
  }

  if (args.decoder !== "internal-wav" && args.decoder !== "ffmpeg") {
    throw new Error("--decoder must be internal-wav or ffmpeg");
  }

  if (args.maxNewTokens !== undefined) {
    const maxNewTokens = Number.parseInt(args.maxNewTokens, 10);
    if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens <= 0) {
      throw new Error("--max-new-tokens must be a positive integer");
    }
    args.maxNewTokens = maxNewTokens;
  }

  return args;
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function readManifest(manifestPath) {
  const manifestDir = dirname(manifestPath);
  return readFileSync(manifestPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const item = JSON.parse(line);
      if (!item.id || !item.audio) {
        throw new Error(`Manifest line ${index + 1} must include id and audio`);
      }
      return {
        ...item,
        audioPath: isAbsolute(item.audio) ? item.audio : resolve(manifestDir, item.audio),
      };
    });
}

function inferWavDurationMs(path) {
  if (!path.toLowerCase().endsWith(".wav")) {
    return undefined;
  }

  const bytes = readFileSync(path);
  if (bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") {
    return undefined;
  }

  let offset = 12;
  let byteRate;
  let dataBytes;

  while (offset + 8 <= bytes.length) {
    const id = bytes.toString("ascii", offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4);
    const body = offset + 8;

    if (id === "fmt ") {
      byteRate = bytes.readUInt32LE(body + 8);
    } else if (id === "data") {
      dataBytes = size;
      break;
    }

    offset = body + size + (size % 2);
  }

  return byteRate && dataBytes ? (dataBytes / byteRate) * 1000 : undefined;
}

function peakRssBytes() {
  return process.memoryUsage().rss;
}

function createTranscriberOptions(args, mod) {
  const options = {
    device: args.device,
    dtype: args.dtype,
    localFilesOnly: args.localFilesOnly,
    model: args.model,
    modelPath: args.modelPath,
    requireLocalModel: args.requireLocalModel,
  };

  if (args.cacheDir) {
    options.cacheDir = args.cacheDir;
  }

  if (args.decoder === "ffmpeg") {
    options.audioDecoderBackend = new mod.FfmpegDecoder();
  }

  return options;
}

async function transcribeWarm(mod, args, items, out) {
  const transcriber = new mod.VoxtralTranscriber(createTranscriberOptions(args, mod));
  const loadStartedAt = performance.now();
  await transcriber.load();
  const modelLoadMs = performance.now() - loadStartedAt;

  try {
    for (const item of items) {
      await transcribeOne(transcriber, args, item, out, modelLoadMs);
    }
  } finally {
    await transcriber.dispose();
  }
}

async function transcribeCold(mod, args, items, out) {
  for (const item of items) {
    const transcriber = new mod.VoxtralTranscriber(createTranscriberOptions(args, mod));
    const loadStartedAt = performance.now();
    await transcriber.load();
    const modelLoadMs = performance.now() - loadStartedAt;

    try {
      await transcribeOne(transcriber, args, item, out, modelLoadMs);
    } finally {
      await transcriber.dispose();
    }
  }
}

async function transcribeOne(transcriber, args, item, out, modelLoadMs) {
  const startedAt = performance.now();
  let record;

  try {
    const result = await transcriber.transcribeFile(item.audioPath, {
      maxNewTokens: args.maxNewTokens,
    });
    const durationMs = performance.now() - startedAt;
    const audioDurationMs = item.durationMs ?? inferWavDurationMs(item.audioPath);

    record = {
      engine: "voxtral-transcribe-ts",
      engineVersion: PACKAGE_JSON.version,
      id: item.id,
      audio: item.audio,
      language: item.language,
      reference: item.text,
      hypothesis: result.text,
      durationMs,
      audioDurationMs,
      rtf: audioDurationMs ? durationMs / audioDurationMs : undefined,
      model: result.model,
      device: args.device,
      dtype: args.dtype,
      decoder: result.decoder,
      mode: args.mode,
      modelLoadMs,
      peakRssBytes: peakRssBytes(),
    };
  } catch (error) {
    record = {
      engine: "voxtral-transcribe-ts",
      engineVersion: PACKAGE_JSON.version,
      id: item.id,
      audio: item.audio,
      language: item.language,
      reference: item.text,
      error: error instanceof Error ? error.message : String(error),
      mode: args.mode,
      modelLoadMs,
      peakRssBytes: peakRssBytes(),
    };
  }

  out.write(`${JSON.stringify(record)}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestPath = resolve(args.manifest);
  const outPath = resolve(args.out);
  const distPath = resolve(ROOT, "dist/index.node.js");

  if (!existsSync(distPath)) {
    throw new Error("Missing dist/index.node.js. Run `npm run build` before the benchmark.");
  }

  const mod = await import(distPath);
  const items = readManifest(manifestPath);
  mkdirSync(dirname(outPath), { recursive: true });
  const out = createWriteStream(outPath, { encoding: "utf8" });

  if (args.mode === "cold") {
    await transcribeCold(mod, args, items, out);
  } else {
    await transcribeWarm(mod, args, items, out);
  }

  await new Promise((resolveWrite, rejectWrite) => {
    out.end((error) => {
      if (error) {
        rejectWrite(error);
      } else {
        resolveWrite();
      }
    });
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
