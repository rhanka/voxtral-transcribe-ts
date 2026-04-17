#!/usr/bin/env node

import { createWriteStream, existsSync, readFileSync } from "node:fs";
import { dirname, extname, isAbsolute, relative, resolve } from "node:path";

function parseArgs(argv) {
  const args = {
    audioTemplate: "{id}.wav",
    durationColumn: "durationMs",
    languageColumn: "language",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }

    const key = toCamelCase(arg.slice(2));
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    args[key] = value;
    i += 1;
  }

  for (const key of ["transcripts", "out", "idColumn", "textColumn"]) {
    if (!args[key]) {
      throw new Error(`Missing --${toKebabCase(key)}`);
    }
  }

  if (!args.audioColumn && !args.audioDir) {
    throw new Error("Provide --audio-column, --audio-dir, or both");
  }

  return args;
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function toKebabCase(value) {
  return value.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

function readRows(path) {
  const text = readFileSync(path, "utf8");
  const extension = extname(path).toLowerCase();

  if (extension === ".jsonl") {
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }

  const delimiter = extension === ".tsv" ? "\t" : ",";
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length === 0) {
    return [];
  }

  const headers = parseDelimitedLine(lines[0], delimiter);
  return lines.slice(1).map((line) => {
    const values = parseDelimitedLine(line, delimiter);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function parseDelimitedLine(line, delimiter) {
  const values = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      quoted = !quoted;
      continue;
    }

    if (char === delimiter && !quoted) {
      values.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function requireColumn(row, column, label) {
  const value = row[column];
  if (value === undefined || value === null || String(value).trim() === "") {
    throw new Error(`Missing ${label} column value: ${column}`);
  }
  return String(value).trim();
}

function resolveAudioPath(row, args) {
  const rawAudio = args.audioColumn
    ? requireColumn(row, args.audioColumn, "audio")
    : args.audioTemplate.replaceAll("{id}", requireColumn(row, args.idColumn, "id"));

  const audioPath = args.audioDir ? resolve(args.audioDir, rawAudio) : resolve(rawAudio);
  if (args.checkFiles && !existsSync(audioPath)) {
    throw new Error(`Audio file does not exist: ${audioPath}`);
  }

  return audioPath;
}

function toManifestRecord(row, args, outDir) {
  const id = requireColumn(row, args.idColumn, "id");
  const text = requireColumn(row, args.textColumn, "text");
  const audioPath = resolveAudioPath(row, args);
  const language = args.language ?? row[args.languageColumn];
  const durationRaw = row[args.durationColumn];

  const record = {
    id,
    audio: normalizeManifestPath(audioPath, outDir),
    text,
  };

  if (language) {
    record.language = String(language).trim();
  }

  if (durationRaw !== undefined && durationRaw !== "") {
    record.durationMs = Number(durationRaw);
  }

  return record;
}

function normalizeManifestPath(path, outDir) {
  const rel = relative(outDir, path);
  if (rel.startsWith("..")) {
    return path;
  }
  return rel.startsWith(".") ? rel : `./${rel}`;
}

async function writeJsonl(path, records) {
  const out = createWriteStream(path, { encoding: "utf8" });
  for (const record of records) {
    out.write(`${JSON.stringify(record)}\n`);
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

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outPath = resolve(args.out);
  const rows = readRows(resolve(args.transcripts));
  const records = rows.map((row) => toManifestRecord(row, args, dirname(outPath)));

  await writeJsonl(outPath, records);
  console.log(`Wrote ${records.length} records to ${outPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
