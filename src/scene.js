'use strict';

const { BLOCK } = require('./codec');

function mulberry32(seed) {
  return function random() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function setPixel(width, mask, gray, x, y, value) {
  if (x < 0 || y < 0 || x >= width || y >= gray.length / width) return;
  const index = y * width + x;
  gray[index] = clamp(value);
  mask[index] = 1;
}

function drawLine(width, mask, gray, x0, y0, x1, y1, value) {
  let dx = Math.abs(x1 - x0);
  let dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let error = dx + dy;
  for (;;) {
    setPixel(width, mask, gray, x0, y0, value);
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * error;
    if (e2 >= dy) {
      error += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      error += dx;
      y0 += sy;
    }
  }
}

function drawRect(width, mask, gray, x, y, w, h, value) {
  drawLine(width, mask, gray, x, y, x + w - 1, y, value);
  drawLine(width, mask, gray, x, y + h - 1, x + w - 1, y + h - 1, value);
  drawLine(width, mask, gray, x, y, x, y + h - 1, value);
  drawLine(width, mask, gray, x + w - 1, y, x + w - 1, y + h - 1, value);
}

function drawGlyph(width, mask, gray, x, y, glyph, value) {
  for (const [x0, y0, x1, y1] of glyph) {
    drawLine(width, mask, gray, x + x0, y + y0, x + x1, y + y1, value);
  }
}

const glyphSet = [
  [[0,0,7,0],[7,0,7,4],[0,4,7,4],[0,4,0,9],[0,9,7,9]],
  [[0,0,7,0],[0,0,0,9],[0,4,7,4],[0,9,7,9],[7,4,7,9]],
  [[0,0,7,0],[0,0,0,4],[0,4,7,4],[7,4,7,9],[0,9,7,9]],
  [[7,0,0,0],[0,0,0,9],[0,5,7,5],[7,5,7,9],[7,9,0,9]],
  [[7,0,0,0],[0,0,0,4],[7,4,0,4],[7,4,7,9],[7,9,0,9]],
  [[1,0,1,9],[1,0,7,0],[1,5,7,5],[7,0,7,9]]
];

const SCROLL_START_X_OFFSET = 126;
const SCROLL_STEP = 2.45;

function makeRoiBlocks(mask, width, height, dilation = 1) {
  const blocksX = width / BLOCK;
  const blocksY = height / BLOCK;
  const roi = new Uint8Array(blocksX * blocksY);
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    const bx = Math.floor(x / BLOCK);
    const by = Math.floor(y / BLOCK);
    for (let oy = -dilation; oy <= dilation; oy += 1) {
      for (let ox = -dilation; ox <= dilation; ox += 1) {
        const nx = bx + ox;
        const ny = by + oy;
        if (nx >= 0 && ny >= 0 && nx < blocksX && ny < blocksY) {
          roi[ny * blocksX + nx] = 1;
        }
      }
    }
  }
  return roi;
}

function createFrame(frameIndex, width, height) {
  const gray = new Uint8Array(width * height);
  const mask = new Uint8Array(width * height);
  const random = mulberry32(9001);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const productTexture =
        118 +
        28 * Math.sin(x * 0.31 + frameIndex * 0.19) +
        22 * Math.sin(y * 0.43 - frameIndex * 0.11) +
        16 * Math.sin((x + y) * 0.17) +
        13 * Math.sin(x * 0.9) * Math.cos(y * 0.73) +
        (random() - 0.5) * 34;
      gray[y * width + x] = clamp(productTexture);
    }
  }

  for (let box = 0; box < 8; box += 1) {
    const x = 24 + box * 39 + Math.round(10 * Math.sin(frameIndex * 0.17 + box));
    const y = 18 + (box % 3) * 47;
    for (let yy = y; yy < y + 28; yy += 2) {
      for (let xx = x; xx < x + 28; xx += 2) {
        if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
          gray[yy * width + xx] = clamp(gray[yy * width + xx] + 42 * ((xx + yy + box) % 3 === 0 ? 1 : -0.6));
        }
      }
    }
  }

  const bandY = 62;
  for (let y = bandY; y < bandY + 36; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const edge = Math.min(y - bandY, bandY + 35 - y);
      const alpha = edge < 2 ? 0.22 : 0.42;
      gray[y * width + x] = clamp(gray[y * width + x] * (1 - alpha) + 232 * alpha);
    }
  }

  const startX = width - SCROLL_START_X_OFFSET - Math.round(frameIndex * SCROLL_STEP);
  const baseline = bandY + 20;
  for (let glyphIndex = 0; glyphIndex < 18; glyphIndex += 1) {
    const x = startX + glyphIndex * 17;
    const y = baseline + (glyphIndex % 2 ? -1 : 0);
    drawGlyph(width, mask, gray, x, y - 10, glyphSet[glyphIndex % glyphSet.length], 26 + (glyphIndex % 3) * 5);
  }

  const glyphCenters = Array.from({ length: 18 }, (_, glyphIndex) => ({
    id: glyphIndex,
    x: startX + glyphIndex * 17 + 4,
    y: baseline - 5
  }));
  return { gray, mask, roiBlocks: makeRoiBlocks(mask, width, height), glyphCenters };
}

module.exports = { createFrame, drawLine, mulberry32, SCROLL_START_X_OFFSET, SCROLL_STEP };
