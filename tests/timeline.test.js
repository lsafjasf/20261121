import assert from "node:assert/strict";
import test from "node:test";
import {
  buildTimeline,
  countInPulses,
  formatClock,
  retimeFuture,
  roundAtTime,
  roundPulses,
} from "../src/timeline.js";

const closeTo = (actual, expected, tolerance = 1e-9) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
};

test("tenth-round pulse lands on the exact accumulated loop boundary", () => {
  const sourceBpm = 92;
  const sourceIn = 2.0;
  const sourceOut = 2.0 + (8 * 60) / sourceBpm;
  const ladder = [72, 78, 84, 90, 96, 102];
  const timeline = buildTimeline({
    anchorTime: 100,
    sourceIn,
    sourceOut,
    sourceBpm,
    ladder,
    countInBars: 1,
    roundCount: 16,
  });

  const countIn = 4 * (60 / ladder[0]);
  let expectedTime = 100 + countIn;
  for (let round = 0; round < 9; round += 1) {
    expectedTime += (8 * 60) / ladder[round % ladder.length];
  }
  const tenth = timeline.rounds[9];

  closeTo(tenth.start, expectedTime);
  closeTo(tenth.pulseStart, 4 + 9 * 8);
  closeTo(tenth.start - timeline.anchorTime, timeline.rounds[0].duration + timeline.rounds.slice(1, 9).reduce((sum, item) => sum + item.duration, countIn));
  closeTo(roundPulses(timeline, 9)[0].time, tenth.start);
  assert.equal(formatClock(tenth.start - 100), formatClock(expectedTime - 100));
});

test("short one-beat loop repeats without cumulative drift", () => {
  const sourceBpm = 140;
  const beatSource = 60 / sourceBpm;
  const timeline = buildTimeline({
    anchorTime: 12.345678,
    sourceIn: 0.25,
    sourceOut: 0.25 + beatSource,
    sourceBpm,
    ladder: [100, 108, 116, 124],
    countInBars: 1,
    roundCount: 128,
  });

  for (let index = 0; index < timeline.rounds.length; index += 1) {
    const round = timeline.rounds[index];
    closeTo(round.duration, 60 / round.bpm);
    closeTo(round.pulseStart, 4 + index);
    closeTo(round.end, round.start + 60 / round.bpm);
  }
  const last = timeline.rounds.at(-1);
  const expected = 12.345678 + 4 * (60 / 100) + Array.from({ length: 128 }, (_, i) => 60 / [100, 108, 116, 124][i % 4]).reduce((sum, value) => sum + value, 0);
  closeTo(last.end, expected);
});

test("hot tempo change preserves past boundaries and retimes the next round", () => {
  const sourceBpm = 120;
  const timeline = buildTimeline({
    anchorTime: 5,
    sourceIn: 1,
    sourceOut: 3,
    sourceBpm,
    ladder: [80, 90, 100, 110],
    countInBars: 1,
    roundCount: 24,
  });
  const originalTwo = timeline.rounds.slice(0, 2).map((round) => ({ ...round }));
  const changed = retimeFuture(timeline, 2, [84, 96, 108, 120, 132]);

  assert.deepEqual(changed.rounds.slice(0, 2), originalTwo);
  closeTo(changed.rounds[2].start, originalTwo[1].end);
  closeTo(changed.rounds[2].bpm, 108);
  closeTo(changed.rounds[2].duration, (2 * 120) / 108);
  closeTo(changed.rounds[3].start, changed.rounds[2].end);
  closeTo(changed.rounds[3].bpm, 120);
  closeTo(changed.rounds[7].bpm, 108);
  assert.equal(changed.rounds.length, timeline.rounds.length);
});

test("count-in and loop pulses remain monotonically ordered", () => {
  const timeline = buildTimeline({
    anchorTime: 20,
    sourceIn: 0,
    sourceOut: 4,
    sourceBpm: 120,
    ladder: [60, 80, 100],
    countInBars: 1,
    roundCount: 12,
  });
  const pulses = [...countInPulses(timeline), ...timeline.rounds.flatMap((_, index) => roundPulses(timeline, index))];
  for (let index = 1; index < pulses.length; index += 1) {
    assert.ok(pulses[index].time > pulses[index - 1].time);
  }
  assert.deepEqual(roundAtTime(timeline, timeline.loopStart).index, 0);
  assert.deepEqual(roundAtTime(timeline, timeline.rounds[11].end - 1e-7).index, 11);
});
