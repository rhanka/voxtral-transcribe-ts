#!/usr/bin/env node

import { createWriteStream, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

function parseArgs(argv) {
  const args = {
    inputs: [],
    out: "bench/results/summary.json",
    stripDiacritics: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--strip-diacritics") {
      args.stripDiacritics = true;
      continue;
    }

    if (arg === "--inputs") {
      i += 1;
      while (i < argv.length && !argv[i].startsWith("--")) {
        args.inputs.push(argv[i]);
        i += 1;
      }
      i -= 1;
      continue;
    }

    if (arg === "--out") {
      args.out = argv[i + 1];
      i += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  if (args.inputs.length === 0) {
    throw new Error("Missing --inputs");
  }

  return args;
}

function readJsonl(path) {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function normalizeText(value, options) {
  let text = String(value ?? "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[’‘`]/g, "'")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (options.stripDiacritics) {
    text = text.normalize("NFD").replace(/\p{M}+/gu, "");
  }

  return text;
}

function levenshtein(a, b) {
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  const current = new Array(b.length + 1);

  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const substitution = previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1);
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, substitution);
    }
    previous.splice(0, previous.length, ...current);
  }

  return previous[b.length];
}

function percentile(values, p) {
  if (values.length === 0) {
    return undefined;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
}

function summarize(records, options) {
  const groups = new Map();

  for (const record of records) {
    const key = [
      record.engine,
      record.engineVersion ?? "unknown",
      record.model ?? "unknown",
      record.device ?? "unknown",
      record.dtype ?? "unknown",
      record.mode ?? "unknown",
    ].join("|");

    if (!groups.has(key)) {
      groups.set(key, {
        engine: record.engine,
        engineVersion: record.engineVersion,
        model: record.model,
        device: record.device,
        dtype: record.dtype,
        mode: record.mode,
        files: 0,
        errors: 0,
        wordDistance: 0,
        referenceWords: 0,
        charDistance: 0,
        referenceChars: 0,
        durationMs: [],
        rtf: [],
        totalAudioDurationMs: 0,
        totalDurationMs: 0,
        peakRssBytes: 0,
        modelLoadMs: [],
      });
    }

    const group = groups.get(key);
    group.files += 1;
    group.peakRssBytes = Math.max(group.peakRssBytes, record.peakRssBytes ?? 0);

    if (typeof record.modelLoadMs === "number") {
      group.modelLoadMs.push(record.modelLoadMs);
    }

    if (record.error) {
      group.errors += 1;
      continue;
    }

    if (typeof record.durationMs === "number") {
      group.durationMs.push(record.durationMs);
      group.totalDurationMs += record.durationMs;
    }

    if (typeof record.audioDurationMs === "number") {
      group.totalAudioDurationMs += record.audioDurationMs;
    }

    if (typeof record.rtf === "number") {
      group.rtf.push(record.rtf);
    }

    if (record.reference && record.hypothesis !== undefined) {
      const reference = normalizeText(record.reference, options);
      const hypothesis = normalizeText(record.hypothesis, options);
      const referenceWords = reference ? reference.split(" ") : [];
      const hypothesisWords = hypothesis ? hypothesis.split(" ") : [];
      const referenceChars = Array.from(reference);
      const hypothesisChars = Array.from(hypothesis);

      group.wordDistance += levenshtein(referenceWords, hypothesisWords);
      group.referenceWords += referenceWords.length;
      group.charDistance += levenshtein(referenceChars, hypothesisChars);
      group.referenceChars += referenceChars.length;
    }
  }

  return [...groups.values()].map((group) => ({
    ...group,
    wer: group.referenceWords > 0 ? group.wordDistance / group.referenceWords : undefined,
    cer: group.referenceChars > 0 ? group.charDistance / group.referenceChars : undefined,
    avgDurationMs: average(group.durationMs),
    p50DurationMs: percentile(group.durationMs, 50),
    p95DurationMs: percentile(group.durationMs, 95),
    avgRtf: average(group.rtf),
    p50Rtf: percentile(group.rtf, 50),
    p95Rtf: percentile(group.rtf, 95),
    avgModelLoadMs: average(group.modelLoadMs),
    throughputAudioPerWall:
      group.totalDurationMs > 0 ? group.totalAudioDurationMs / group.totalDurationMs : undefined,
  }));
}

function average(values) {
  if (values.length === 0) {
    return undefined;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function format(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "n/a";
}

function printMarkdown(summary) {
  console.log("| Engine | Model | Device | Mode | Files | Errors | WER | CER | Avg RTF | P95 RTF | Avg ms | P95 ms | Peak RSS MB |");
  console.log("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const row of summary) {
    console.log(
      [
        row.engine,
        row.model,
        row.device,
        row.mode,
        row.files,
        row.errors,
        format(row.wer),
        format(row.cer),
        format(row.avgRtf),
        format(row.p95Rtf),
        format(row.avgDurationMs, 1),
        format(row.p95DurationMs, 1),
        format(row.peakRssBytes / 1024 / 1024, 1),
      ].join(" | ").replace(/^/, "| ").replace(/$/, " |"),
    );
  }
}

async function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const out = createWriteStream(path, { encoding: "utf8" });
  out.write(`${JSON.stringify(value, null, 2)}\n`);
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

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const records = args.inputs.flatMap((input) => readJsonl(resolve(input)));
  const summary = summarize(records, args);

  printMarkdown(summary);
  await writeJson(resolve(args.out), {
    generatedAt: new Date().toISOString(),
    inputs: args.inputs,
    normalization: {
      lowercase: true,
      punctuationToSpace: true,
      stripDiacritics: args.stripDiacritics,
    },
    summary,
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
