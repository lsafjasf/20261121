import assert from "node:assert/strict";
import test from "node:test";
import { timeStretchChannel } from "../src/stretch.js";

const SR = 44100;

function estimateFrequency(samples, sampleRate) {
  let crossings = 0;
  let first = -1;
  let last = -1;
  for (let index = 1; index < samples.length; index += 1) {
    if (samples[index - 1] < 0 && samples[index] >= 0) {
      crossings += 1;
      if (first < 0) first = index;
      last = index;
    }
  }
  return crossings > 1 ? ((crossings - 1) * sampleRate) / (last - first) : 0;
}

for (const ratio of [0.72, 1.18]) {
  test(`pitch stays at 440 Hz when stretching by ${ratio}`, () => {
    const input = new Float32Array(SR * 2);
    for (let index = 0; index < input.length; index += 1) {
      input[index] = Math.sin((2 * Math.PI * 440 * index) / SR) * 0.7;
    }
    const output = timeStretchChannel(input, ratio, SR);
    assert.equal(output.length, Math.round(input.length / ratio));
    assert.ok(Math.abs(estimateFrequency(output, SR) - 440) < 1.5);
  });
}

test("short material still has a deterministic exact output length", () => {
  const input = Float32Array.from({ length: 257 }, (_, index) => Math.sin(index / 4));
  for (const ratio of [0.5, 0.93, 1, 1.35]) {
    const output = timeStretchChannel(input, ratio, SR, {
      frameSize: 128,
      synthesisHop: 64,
      tolerance: 12,
    });
    assert.equal(output.length, Math.max(1, Math.round(input.length / ratio)));
  }
});
