'use strict';

function labelComponents(mask, width, height, predicate = () => true, minSize = 3) {
  const labels = new Int32Array(mask.length).fill(-1);
  const components = [];
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || labels[start] !== -1 || !predicate(start)) continue;
    const pixels = [];
    const stack = [start];
    labels[start] = components.length;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    while (stack.length) {
      const index = stack.pop();
      pixels.push(index);
      const x = index % width;
      const y = (index / width) | 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      const neighbors = [
        index - width - 1, index - width, index - width + 1,
        index - 1, index + 1,
        index + width - 1, index + width, index + width + 1
      ];
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= mask.length || labels[neighbor] !== -1 || !mask[neighbor]) continue;
        const nx = neighbor % width;
        const ny = (neighbor / width) | 0;
        if (Math.abs(nx - x) > 1 || Math.abs(ny - y) > 1) continue;
        if (!predicate(neighbor)) continue;
        labels[neighbor] = components.length;
        stack.push(neighbor);
      }
    }
    if (pixels.length >= minSize) {
      const cx = pixels.reduce((sum, index) => sum + index % width, 0) / pixels.length;
      const cy = pixels.reduce((sum, index) => sum + (index / width | 0), 0) / pixels.length;
      components.push({ pixels, minX, minY, maxX, maxY, cx, cy });
    }
  }
  return { labels, components };
}

function meanSquared(original, decoded, selector) {
  let sum = 0;
  let count = 0;
  for (let index = 0; index < original.length; index += 1) {
    if (!selector(index)) continue;
    const difference = original[index] - decoded[index];
    sum += difference * difference;
    count += 1;
  }
  return count ? sum / count : 0;
}

function psnr(mse) {
  return mse === 0 ? 99 : 10 * Math.log10((255 * 255) / mse);
}

function analyzeFrame(
  frameIndex,
  original,
  mask,
  decoded,
  width,
  height,
  backgroundMask,
  tracker,
  glyphCenters
) {
  const { components } = labelComponents(mask, width, height, () => true, 4);
  const strokes = [];
  const assignedCenters = new Set();
  for (const component of components) {
    const survivors = component.pixels.filter((index) => decoded[index] <= 95);
    const survivingMask = new Uint8Array(mask.length);
    for (const index of survivors) survivingMask[index] = 1;
    const survivorComponents = labelComponents(survivingMask, width, height, () => true, 1).components;
    const largestSurvivor = survivorComponents.reduce((max, item) => Math.max(max, item.pixels.length), 0);
    const survival = survivors.length / component.pixels.length;
    const continuity = component.pixels.length ? largestSurvivor / component.pixels.length : 0;
    const lost = 1 - survival;
    const broken = survivors.length === 0 || continuity < 0.62;

    let bestId = -1;
    let bestDistance = Infinity;
    for (const center of glyphCenters) {
      if (assignedCenters.has(center.id)) continue;
      const distance = Math.hypot(center.x - component.cx, center.y - component.cy);
      if (distance < bestDistance && distance < 26) {
        bestDistance = distance;
        bestId = center.id;
      }
    }
    if (bestId === -1) bestId = 1000 + tracker.nextId++;
    assignedCenters.add(bestId);
    if (!tracker.records.has(bestId)) {
      tracker.records.set(bestId, { run: 0, longest: 0 });
    }
    const record = tracker.records.get(bestId);
    record.run = broken ? record.run + 1 : 0;
    record.longest = Math.max(record.longest, record.run);

    strokes.push({
      id: bestId,
      frame: frameIndex,
      x: Math.round(component.cx),
      y: Math.round(component.cy),
      bbox: [component.minX, component.minY, component.maxX, component.maxY],
      pixels: component.pixels.length,
      survivors: survivors.length,
      lost,
      continuity,
      broken,
      longestMissingFrames: record.longest
    });
  }

  const textMse = meanSquared(original, decoded, (index) => mask[index]);
  const backgroundMse = meanSquared(original, decoded, (index) => backgroundMask[index]);
  return {
    strokes,
    textMse,
    textPsnr: psnr(textMse),
    backgroundMse,
    backgroundPsnr: psnr(backgroundMse),
    lostTextPixels: strokes.reduce((sum, stroke) => sum + stroke.pixels - stroke.survivors, 0),
    brokenStrokes: strokes.filter((stroke) => stroke.broken).length
  };
}

module.exports = { analyzeFrame, labelComponents, meanSquared, psnr };
