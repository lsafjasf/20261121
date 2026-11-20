'use strict';

const zlib = require('node:zlib');

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const value of buffer) {
    crc = crcTable[(crc ^ value) & 255] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function u32(value) {
  return [
    (value >>> 24) & 255,
    (value >>> 16) & 255,
    (value >>> 8) & 255,
    value & 255
  ];
}

function chunk(type, data) {
  const typeBytes = Buffer.from(type, 'ascii');
  const length = Buffer.from(u32(data.length));
  const body = Buffer.concat([typeBytes, data]);
  const checksum = Buffer.from(u32(crc32(body)));
  return Buffer.concat([length, body, checksum]);
}

function rawScanlines(width, channels, data) {
  const stride = width * channels;
  const output = Buffer.alloc((stride + 1) * (data.length / stride));
  for (let y = 0, src = 0, dst = 0; src < data.length; y += 1, src += stride) {
    output[dst] = 0;
    data.copy(output, dst + 1, src, src + stride);
    dst += stride + 1;
  }
  return output;
}

function encodePng(width, height, data, colorType) {
  const channels = colorType === 0 ? 1 : 3;
  if (data.length !== width * height * channels) {
    throw new RangeError('PNG data dimensions do not match');
  }
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.from([
    ...u32(width),
    ...u32(height),
    8,
    colorType,
    0,
    0,
    0
  ]);
  const idat = zlib.deflateSync(rawScanlines(width, channels, data), {
    level: 9,
    memLevel: 9
  });
  return Buffer.concat([
    signature,
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

function encodeGray(width, height, gray) {
  return encodePng(width, height, Buffer.from(gray), 0);
}

function encodeRgb(width, height, rgb) {
  return encodePng(width, height, Buffer.from(rgb), 2);
}

module.exports = { encodeGray, encodeRgb };
