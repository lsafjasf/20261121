'use strict';

const BLOCK = 8;

const zigzag = [
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63
];

const dctMatrix = (() => {
  const matrix = Array.from({ length: BLOCK }, () => new Float64Array(BLOCK));
  for (let u = 0; u < BLOCK; u += 1) {
    const alpha = u === 0 ? Math.sqrt(1 / BLOCK) : Math.sqrt(2 / BLOCK);
    for (let x = 0; x < BLOCK; x += 1) {
      matrix[u][x] = alpha * Math.cos(((2 * x + 1) * u * Math.PI) / 16);
    }
  }
  return matrix;
})();

const frequencyFactor = new Float32Array(64);
for (let scan = 0; scan < 64; scan += 1) {
  frequencyFactor[zigzag[scan]] = 1 + scan * 0.13;
}

class ByteWriter {
  constructor() {
    this.parts = [];
  }

  byte(value) {
    this.parts.push(Buffer.from([value & 255]));
  }

  u16(value) {
    this.parts.push(Buffer.from([(value >>> 8) & 255, value & 255]));
  }

  uleb(value) {
    let current = Math.max(0, Math.trunc(value));
    do {
      let next = current & 127;
      current >>>= 7;
      if (current !== 0) next |= 128;
      this.byte(next);
    } while (current !== 0);
  }

  signed(value) {
    this.uleb(((value << 1) ^ (value >> 31)) >>> 0);
  }

  buffer() {
    return Buffer.concat(this.parts);
  }
}

class ByteReader {
  constructor(buffer) {
    this.buffer = buffer;
    this.offset = 0;
  }

  byte() {
    if (this.offset >= this.buffer.length) throw new RangeError('Truncated frame');
    return this.buffer[this.offset++];
  }

  u16() {
    return (this.byte() << 8) | this.byte();
  }

  uleb() {
    let value = 0;
    let shift = 0;
    let current;
    do {
      current = this.byte();
      value |= (current & 127) << shift;
      shift += 7;
    } while (current & 128);
    return value >>> 0;
  }

  signed() {
    const value = this.uleb();
    return (value >>> 1) ^ -(value & 1);
  }
}

function blockCount(width, height) {
  return (width / BLOCK) * (height / BLOCK);
}

function transformFrame(gray, width, height) {
  const count = blockCount(width, height);
  const coefficients = new Float32Array(count * 64);
  for (let by = 0, block = 0; by < height; by += BLOCK) {
    for (let bx = 0; bx < width; bx += BLOCK, block += 1) {
      const pixels = new Float64Array(64);
      for (let y = 0; y < BLOCK; y += 1) {
        for (let x = 0; x < BLOCK; x += 1) {
          pixels[y * BLOCK + x] = gray[(by + y) * width + bx + x] - 128;
        }
      }
      const temp = new Float64Array(64);
      for (let u = 0; u < BLOCK; u += 1) {
        for (let y = 0; y < BLOCK; y += 1) {
          let sum = 0;
          for (let x = 0; x < BLOCK; x += 1) sum += dctMatrix[u][x] * pixels[y * BLOCK + x];
          temp[u * BLOCK + y] = sum;
        }
      }
      const output = coefficients.subarray(block * 64, block * 64 + 64);
      for (let u = 0; u < BLOCK; u += 1) {
        for (let v = 0; v < BLOCK; v += 1) {
          let sum = 0;
          for (let y = 0; y < BLOCK; y += 1) {
            sum += temp[u * BLOCK + y] * dctMatrix[v][y];
          }
          output[u * BLOCK + v] = sum;
        }
      }
    }
  }
  return coefficients;
}

function quantizeBlock(coefficients, block, q, writer, previousDc, ordered) {
  const base = block * 64;
  const quantized = new Int32Array(64);
  for (let scan = 0; scan < 64; scan += 1) {
    const index = zigzag[scan];
    const step = q * frequencyFactor[index];
    quantized[index] = Math.round(coefficients[base + index] / step);
  }
  writer.signed(quantized[0] - previousDc);
  const nonZero = [];
  for (let scan = 1; scan < 64; scan += 1) {
    const value = quantized[zigzag[scan]];
    if (value !== 0) nonZero.push([scan, value]);
  }
  writer.uleb(nonZero.length);
  for (const [scan, value] of nonZero) {
    writer.uleb(scan);
    writer.signed(value);
  }
  ordered.push(quantized);
  return quantized[0];
}

function encodeFrame(gray, width, height, blockQuantizers, roiBlocks) {
  const count = blockCount(width, height);
  if (blockQuantizers.length !== count || roiBlocks.length !== count) {
    throw new RangeError('Block quantizer arrays have invalid dimensions');
  }
  const coefficients = transformFrame(gray, width, height);
  const writer = new ByteWriter();
  writer.u16(width);
  writer.u16(height);
  const roiList = [];
  for (let block = 0; block < count; block += 1) {
    if (roiBlocks[block]) roiList.push(block);
  }
  writer.uleb(roiList.length);

  let run = 0;
  for (let block = 0; block < count; block += 1) {
    if (roiBlocks[block]) {
      if (run > 0) {
        writer.uleb(0);
        writer.uleb(run);
        run = 0;
      }
      writer.uleb(block + 1);
    } else {
      run += 1;
    }
  }
  if (run > 0) {
    writer.uleb(0);
    writer.uleb(run);
  }
  writer.uleb(0);
  writer.uleb(0);
  const uniqueQ = [...new Set(blockQuantizers.map((value) => Math.round(value)))];
  writer.byte(uniqueQ.length === 1 ? 0 : 1);
  for (const value of uniqueQ) writer.uleb(value);

  let previousDc = 0;
  const ordered = [];
  for (let block = 0; block < count; block += 1) {
    previousDc = quantizeBlock(
      coefficients,
      block,
      blockQuantizers[block],
      writer,
      previousDc,
      ordered
    );
  }
  return { bytes: writer.buffer(), coefficients, quantized: ordered };
}

function dequantizeBlock(quantized, q) {
  const output = new Float64Array(64);
  for (let index = 0; index < 64; index += 1) {
    output[index] = quantized[index] * q * frequencyFactor[index];
  }
  return output;
}

function inverseTransform(coefficients, width, height, x0, y0, output) {
  for (let y = 0; y < BLOCK; y += 1) {
    for (let x = 0; x < BLOCK; x += 1) {
      let sum = 0;
      for (let u = 0; u < BLOCK; u += 1) {
        for (let v = 0; v < BLOCK; v += 1) {
          sum += dctMatrix[u][x] * coefficients[u * BLOCK + v] * dctMatrix[v][y];
        }
      }
      const value = Math.round(sum + 128);
      output[(y0 + y) * width + x0 + x] = Math.max(0, Math.min(255, value));
    }
  }
}

function decodeFrame(buffer) {
  const reader = new ByteReader(buffer);
  const width = reader.u16();
  const height = reader.u16();
  const count = blockCount(width, height);
  const roiBlocks = new Uint8Array(count);
  const roiItems = reader.uleb();
  for (;;) {
    const marker = reader.uleb();
    if (marker === 0) {
      const length = reader.uleb();
      if (length === 0) break;
    } else {
      const block = marker - 1;
      if (block < 0 || block >= count) throw new RangeError('Invalid ROI block');
      roiBlocks[block] = 1;
    }
  }
  let roiCheck = 0;
  for (const value of roiBlocks) roiCheck += value;
  if (roiCheck !== roiItems) throw new RangeError('Invalid ROI header');
  const mode = reader.byte();
  const qValues = [];
  qValues.push(reader.uleb());
  if (mode === 1) qValues.push(reader.uleb());
  const quantizers = new Float32Array(count);
  const output = new Uint8Array(width * height);
  let previousDc = 0;
  for (let block = 0; block < count; block += 1) {
    const dc = reader.signed() + previousDc;
    previousDc = dc;
    const quantized = new Int32Array(64);
    quantized[0] = dc;
    const nonZeroCount = reader.uleb();
    if (nonZeroCount > 63) throw new RangeError('Invalid AC coefficient count');
    for (let item = 0; item < nonZeroCount; item += 1) {
      const scan = reader.uleb();
      if (scan < 1 || scan > 63) throw new RangeError('Invalid AC coefficient position');
      quantized[zigzag[scan]] = reader.signed();
    }
    quantizers[block] = mode === 0 ? qValues[0] : qValues[roiBlocks[block]];
    const by = Math.floor(block / (width / BLOCK));
    const bx = block % (width / BLOCK);
    inverseTransform(
      dequantizeBlock(quantized, quantizers[block]),
      width,
      height,
      bx * BLOCK,
      by * BLOCK,
      output
    );
  }
  return { gray: output, width, height, roiBlocks, quantizers };
}

function uniformQuantizers(count, q) {
  return new Float32Array(count).fill(q);
}

function roiQuantizers(count, roiBlocks, textQ, backgroundQ) {
  const values = new Float32Array(count);
  for (let block = 0; block < count; block += 1) {
    values[block] = roiBlocks[block] ? textQ : backgroundQ;
  }
  return values;
}

function fitUniform(gray, width, height, targetBytes, roiBlocks, low = 1, high = 48) {
  let best = null;
  while (low <= high) {
    const q = Math.floor((low + high) / 2);
    const count = blockCount(width, height);
    const result = encodeFrame(gray, width, height, uniformQuantizers(count, q), roiBlocks);
    const candidate = { q, bytes: result.bytes, result };
    if (!best || Math.abs(result.bytes.length - targetBytes) < Math.abs(best.bytes.length - targetBytes)) {
      best = candidate;
    }
    if (result.bytes.length > targetBytes) low = q + 1;
    else high = q - 1;
  }
  return best;
}

function fitRoi(gray, width, height, targetBytes, roiBlocks, textQ = 5, low = 1, high = 48) {
  let best = null;
  while (low <= high) {
    const backgroundQ = Math.floor((low + high) / 2);
    const count = blockCount(width, height);
    const result = encodeFrame(
      gray,
      width,
      height,
      roiQuantizers(count, roiBlocks, textQ, backgroundQ),
      roiBlocks
    );
    const candidate = { textQ, backgroundQ, bytes: result.bytes, result };
    if (!best) {
      best = candidate;
    } else if (
      result.bytes.length <= targetBytes &&
      (best.bytes.length > targetBytes || result.bytes.length > best.bytes.length)
    ) {
      best = candidate;
    } else if (
      best.bytes.length > targetBytes &&
      result.bytes.length > targetBytes &&
      result.bytes.length < best.bytes.length
    ) {
      best = candidate;
    }
    if (result.bytes.length > targetBytes) low = backgroundQ + 1;
    else high = backgroundQ - 1;
  }
  return best;
}

function fitAdaptiveRoi(gray, width, height, targetBytes, roiBlocks) {
  let fallback = null;
  for (let textQ = 1; textQ <= 32; textQ += 1) {
    const candidate = fitRoi(gray, width, height, targetBytes, roiBlocks, textQ);
    if (!fallback || candidate.bytes.length < fallback.bytes.length) fallback = candidate;
    if (candidate.bytes.length <= targetBytes) return candidate;
  }
  return fallback;
}

module.exports = {
  BLOCK,
  blockCount,
  decodeFrame,
  encodeFrame,
  fitAdaptiveRoi,
  fitRoi,
  fitUniform,
  roiQuantizers,
  transformFrame,
  uniformQuantizers
};
