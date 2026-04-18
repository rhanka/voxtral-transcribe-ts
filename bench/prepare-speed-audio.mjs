#!/usr/bin/env node

import { createWriteStream, mkdirSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { pipeline } from "node:stream/promises";

const DEFAULT_URL =
  "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/benchmark/benchmark.m4a";

function parseArgs(argv) {
  const args = {
    durationMs: 13 * 60 * 1000,
    id: "faster_whisper_benchmark_m4a",
    language: "fr",
    outAudio: "bench/datasets/faster-whisper-speed/benchmark.m4a",
    outManifest: "bench/datasets/faster-whisper-speed.jsonl",
    text: "",
    url: DEFAULT_URL,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }

    const key = arg.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    args[key] = value;
    i += 1;
  }

  args.durationMs = Number(args.durationMs);
  return args;
}

async function download(url, outPath) {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  mkdirSync(dirname(outPath), { recursive: true });
  await pipeline(response.body, createWriteStream(outPath));
}

async function writeManifest(args) {
  const outManifest = resolve(args.outManifest);
  const outAudio = resolve(args.outAudio);
  const relAudio = relativePath(dirname(outManifest), outAudio);
  const record = {
    id: args.id,
    audio: relAudio,
    text: args.text,
    language: args.language,
    durationMs: args.durationMs,
    benchmarkPurpose: "speed",
    source: "SYSTRAN/faster-whisper benchmark/benchmark.m4a",
    sourceUrl: args.url,
  };

  mkdirSync(dirname(outManifest), { recursive: true });
  const out = createWriteStream(outManifest, { encoding: "utf8" });
  out.write(`${JSON.stringify(record)}\n`);
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

function relativePath(fromDir, toPath) {
  const rel = relative(fromDir, toPath);
  if (rel.startsWith("..")) {
    return toPath;
  }
  return rel.startsWith(".") ? rel : `./${rel}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outAudio = resolve(args.outAudio);

  await download(args.url, outAudio);
  await writeManifest(args);
  console.log(`Downloaded ${args.url}`);
  console.log(`Audio: ${outAudio}`);
  console.log(`Manifest: ${resolve(args.outManifest)}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
