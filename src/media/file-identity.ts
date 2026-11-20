import type { MediaIdentity } from "../core/types.ts";

const SAMPLE_BYTES = 512 * 1024;

function toHex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function joinBytes(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const length = parts.reduce((total, part) => total + part.byteLength, 0);
  const output = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

export async function computeMediaIdentity(file: File): Promise<MediaIdentity> {
  const firstLength = Math.min(SAMPLE_BYTES, file.size);
  const lastStart = Math.max(0, file.size - SAMPLE_BYTES);
  const [first, last] = await Promise.all([
    file.slice(0, firstLength).arrayBuffer(),
    file.slice(lastStart).arrayBuffer(),
  ]);

  const metadata = new TextEncoder().encode(
    `${file.name}\0${file.size}\0${file.lastModified}\0${firstLength}\0${last.byteLength}\0`,
  );
  const digest = await crypto.subtle.digest(
    "SHA-256",
    joinBytes([metadata, new Uint8Array(first), new Uint8Array(last)]),
  );
  const hash = toHex(digest);

  return {
    id: `sha256:${hash}`,
    name: file.name,
    sizeBytes: file.size,
    lastModified: file.lastModified,
    fingerprint: `sha256-sampled:${hash}:${firstLength}+${last.byteLength}`,
  };
}
