'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {
  BLOCK,
  decodeFrame,
  fitAdaptiveRoi,
  fitUniform
} = require('./codec');
const { createFrame } = require('./scene');
const { analyzeFrame } = require('./analysis');
const { encodeRgb } = require('./png');
const { buildReportHtml } = require('./report');

const WIDTH = 320;
const HEIGHT = 176;
const FRAME_COUNT = 48;
const FPS = 12;
const TARGET_BYTES = 11500;
const OUT = path.join(__dirname, '..', 'artifacts', 'experiment');

function rgbFromGray(gray) {
  const rgb = new Uint8Array(gray.length * 3);
  for (let index = 0; index < gray.length; index += 1) {
    rgb[index * 3] = gray[index];
    rgb[index * 3 + 1] = gray[index];
    rgb[index * 3 + 2] = gray[index];
  }
  return rgb;
}

function overlay(rgb, mask, color) {
  const output = new Uint8Array(rgb);
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const target = index * 3;
    output[target] = color[0];
    output[target + 1] = color[1];
    output[target + 2] = color[2];
  }
  return output;
}

function buildErrorPanel(original, decoded, mask, uniformBrokenPixels, roiBrokenPixels, roiBlocks) {
  const blocksX = WIDTH / BLOCK;
  const rgb = rgbFromGray(original);
  for (let index = 0; index < original.length; index += 1) {
    const target = index * 3;
    const difference = Math.abs(original[index] - decoded[index]);
    if (roiBrokenPixels[index]) {
      rgb[target] = 0;
      rgb[target + 1] = 230;
      rgb[target + 2] = 118;
    } else if (uniformBrokenPixels[index]) {
      rgb[target] = 255;
      rgb[target + 1] = 238;
      rgb[target + 2] = 0;
    } else if (mask[index]) {
      rgb[target] = Math.min(255, 80 + difference * 4);
      rgb[target + 1] = 90;
      rgb[target + 2] = 255;
    } else if (difference > 18) {
      rgb[target] = Math.min(255, difference * 5);
      rgb[target + 1] = 40;
      rgb[target + 2] = 40;
    } else {
      const block = (Math.floor(index / WIDTH) / BLOCK | 0) * blocksX + (index % WIDTH) / BLOCK;
      if (roiBlocks[block]) {
        rgb[target] = 28;
        rgb[target + 1] = 76;
        rgb[target + 2] = 64;
      } else {
        rgb[target] = 28;
        rgb[target + 1] = 28;
        rgb[target + 2] = 34;
      }
    }
  }
  return rgb;
}

function writePanel(name, panels) {
  const gutter = 6;
  const width = panels.length * WIDTH + (panels.length - 1) * gutter;
  const output = new Uint8Array(width * HEIGHT * 3);
  for (let panel = 0; panel < panels.length; panel += 1) {
    const offsetX = panel * (WIDTH + gutter);
    for (let y = 0; y < HEIGHT; y += 1) {
      for (let x = 0; x < WIDTH; x += 1) {
        const source = (y * WIDTH + x) * 3;
        const target = (y * width + offsetX + x) * 3;
        output[target] = panels[panel][source];
        output[target + 1] = panels[panel][source + 1];
        output[target + 2] = panels[panel][source + 2];
      }
    }
  }
  return fs.writeFile(path.join(OUT, 'frames', name), encodeRgb(width, HEIGHT, output));
}

function markBrokenPixels(metrics, mask, decoded, output) {
  for (const stroke of metrics.strokes) {
    if (!stroke.broken) continue;
    const [minX, minY, maxX, maxY] = stroke.bbox;
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const index = y * WIDTH + x;
        if (mask[index] && decoded[index] > 95) output[index] = 1;
      }
    }
  }
}

async function main() {
  await fs.rm(OUT, { recursive: true, force: true });
  await fs.mkdir(path.join(OUT, 'frames'), { recursive: true });
  const frames = [];
  const uniformFrames = [];
  const roiFrames = [];
  const uniformTracker = { nextId: 1, records: new Map() };
  const roiTracker = { nextId: 1, records: new Map() };
  let uniformTotal = 0;
  let roiTotal = 0;

  for (let frameIndex = 0; frameIndex < FRAME_COUNT; frameIndex += 1) {
    const scene = createFrame(frameIndex, WIDTH, HEIGHT);
    const uniform = fitUniform(scene.gray, WIDTH, HEIGHT, TARGET_BYTES, scene.roiBlocks);
    const targetBytes = uniform.bytes.length;
    const roi = fitAdaptiveRoi(scene.gray, WIDTH, HEIGHT, targetBytes, scene.roiBlocks);

    const uniformDecoded = decodeFrame(uniform.bytes);
    const roiDecoded = decodeFrame(roi.bytes);
    const blocksX = WIDTH / BLOCK;
    const backgroundMask = new Uint8Array(scene.mask.length);
    for (let index = 0; index < scene.mask.length; index += 1) {
      const x = index % WIDTH;
      const y = Math.floor(index / WIDTH);
      const block = Math.floor(y / BLOCK) * blocksX + Math.floor(x / BLOCK);
      backgroundMask[index] = scene.roiBlocks[block] ? 0 : 1;
    }

    const uniformMetrics = analyzeFrame(
      frameIndex,
      scene.gray,
      scene.mask,
      uniformDecoded.gray,
      WIDTH,
      HEIGHT,
      backgroundMask,
      uniformTracker,
      scene.glyphCenters
    );
    const roiMetrics = analyzeFrame(
      frameIndex,
      scene.gray,
      scene.mask,
      roiDecoded.gray,
      WIDTH,
      HEIGHT,
      backgroundMask,
      roiTracker,
      scene.glyphCenters
    );

    const uniformBrokenPixels = new Uint8Array(scene.mask.length);
    const roiBrokenPixels = new Uint8Array(scene.mask.length);
    markBrokenPixels(uniformMetrics, scene.mask, uniformDecoded.gray, uniformBrokenPixels);
    markBrokenPixels(roiMetrics, scene.mask, roiDecoded.gray, roiBrokenPixels);

    const originalPanel = overlay(rgbFromGray(scene.gray), scene.mask, [80, 220, 255]);
    const uniformPanel = rgbFromGray(uniformDecoded.gray);
    const roiPanel = rgbFromGray(roiDecoded.gray);
    const errorPanel = buildErrorPanel(
      scene.gray,
      uniformDecoded.gray,
      scene.mask,
      uniformBrokenPixels,
      roiBrokenPixels,
      scene.roiBlocks
    );
    await writePanel(`frame_${String(frameIndex).padStart(3, '0')}.png`, [
      originalPanel,
      uniformPanel,
      roiPanel,
      errorPanel
    ]);

    const frame = {
      frame: frameIndex,
      timeSeconds: Number((frameIndex / FPS).toFixed(3)),
      targetBytes,
      uniform: {
        bytes: uniform.bytes.length,
        q: uniform.q,
        ...uniformMetrics
      },
      roi: {
        bytes: roi.bytes.length,
        textQ: roi.textQ,
        backgroundQ: roi.backgroundQ,
        ...roiMetrics
      }
    };
    frames.push(frame);
    uniformTotal += uniform.bytes.length;
    roiTotal += roi.bytes.length;
    uniformFrames.push(uniform.bytes);
    roiFrames.push(roi.bytes);
  }

  const summary = summarize(frames, uniformTotal, roiTotal);
  const metrics = {
    config: {
      width: WIDTH,
      height: HEIGHT,
      frames: FRAME_COUNT,
      fps: FPS,
      targetBytesPerFrame: TARGET_BYTES,
      textQ: 'adaptive 1-32',
      dctBlock: BLOCK
    },
    summary,
    frames
  };
  await fs.writeFile(path.join(OUT, 'metrics.json'), JSON.stringify(metrics, null, 2));
  await writeContainer(path.join(OUT, 'uniform.bin'), uniformFrames);
  await writeContainer(path.join(OUT, 'roi.bin'), roiFrames);
  await fs.writeFile(path.join(OUT, 'report.html'), buildReportHtml(metrics));
  console.log(JSON.stringify(summary, null, 2));
}

async function writeContainer(file, frameBuffers) {
  const header = Buffer.alloc(6);
  header.write('TRDCT', 0, 'ascii');
  header.writeUInt8(frameBuffers.length, 5);
  const lengths = Buffer.concat(frameBuffers.map((frame) => {
    const entry = Buffer.alloc(4);
    entry.writeUInt32BE(frame.length, 0);
    return entry;
  }));
  await fs.writeFile(file, Buffer.concat([header, lengths, ...frameBuffers]));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarize(frames, uniformTotal, roiTotal) {
  const modes = ['uniform', 'roi'];
  const summary = {
    uniformBytes: uniformTotal,
    roiBytes: roiTotal,
    byteDelta: roiTotal - uniformTotal,
    byteRatio: roiTotal / uniformTotal
  };
  for (const mode of modes) {
    summary[mode] = {
      avgBytes: average(frames.map((frame) => frame[mode].bytes)),
      avgTextPsnr: average(frames.map((frame) => frame[mode].textPsnr)),
      avgBackgroundPsnr: average(frames.map((frame) => frame[mode].backgroundPsnr)),
      totalLostTextPixels: frames.reduce((sum, frame) => sum + frame[mode].lostTextPixels, 0),
      totalBrokenStrokes: frames.reduce((sum, frame) => sum + frame[mode].brokenStrokes, 0),
      longestMissingSeconds: Math.max(...frames.flatMap((frame) => frame[mode].strokes.map((stroke) => stroke.longestMissingFrames))) / FPS,
      worstBackgroundFrame: frames.toSorted((a, b) => b[mode].backgroundMse - a[mode].backgroundMse)[0].frame
    };
  }
  return summary;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
