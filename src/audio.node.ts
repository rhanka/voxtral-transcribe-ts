import { readFile } from "node:fs/promises";
export { decodeWav, resampleAudio, type DecodedWav } from "./audio-core.js";
import { decodeWav, type DecodedWav } from "./audio-core.js";

export async function readWavFile(path: string | URL): Promise<DecodedWav> {
  const buffer = await readFile(path);
  return decodeWav(buffer);
}
