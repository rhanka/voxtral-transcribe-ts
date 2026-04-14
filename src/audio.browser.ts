export { decodeWav, resampleAudio, type DecodedWav } from "./audio-core.js";
import { decodeWav, type DecodedWav } from "./audio-core.js";

function isBlobLike(value: unknown): value is Blob {
  return typeof Blob !== "undefined" && value instanceof Blob;
}

export async function readWavFile(input: Blob | string | URL): Promise<DecodedWav> {
  if (isBlobLike(input)) {
    return decodeWav(new Uint8Array(await input.arrayBuffer()));
  }

  if (typeof fetch !== "function") {
    throw new Error("readWavFile requires fetch in browser environments.");
  }

  const response = await fetch(input.toString());
  if (!response.ok) {
    throw new Error(`Failed to fetch WAV input "${input.toString()}": ${response.status} ${response.statusText}`);
  }

  return decodeWav(new Uint8Array(await response.arrayBuffer()));
}
