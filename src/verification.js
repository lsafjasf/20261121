import { buildTimeline, retimeFuture } from "./timeline.js";

const near = (actual, expected, tolerance = 1e-6) => Math.abs(actual - expected) <= tolerance;

export function runTimingChecks({ sourceBpm = 92, sourceIn = 0, sourceOut = 8 * 60 / 92, ladder = [70, 76, 82, 88, 94, 100, 106, 112] } = {}) {
  const loopBeats = ((sourceOut - sourceIn) * sourceBpm) / 60;

  const tenthTimeline = buildTimeline({
    anchorTime: 0,
    sourceIn,
    sourceOut,
    sourceBpm,
    ladder,
    countInBars: 1,
    roundCount: Math.max(12, ladder.length),
  });
  const tenth = tenthTimeline.rounds[9];
  const tenthExpected = tenthTimeline.loopStart + tenthTimeline.rounds
    .slice(0, 9)
    .reduce((sum, round) => sum + round.duration, 0);
  const tenthPulseExpected = tenthTimeline.countInBeats + 9 * loopBeats;
  const tenthError = Math.abs(tenth.start - tenthExpected);
  const tenthPulseError = Math.abs(tenth.pulseStart - tenthPulseExpected);

  const shortBeat = 60 / sourceBpm;
  const shortTimeline = buildTimeline({
    anchorTime: 3.14159,
    sourceIn,
    sourceOut: sourceIn + shortBeat,
    sourceBpm,
    ladder: [96, 104, 112, 120],
    countInBars: 1,
    roundCount: 32,
  });
  const shortExpected = Array.from({ length: 32 }, (_, index) => shortTimeline.rounds[index].start + 60 / shortTimeline.rounds[index].bpm);
  const shortError = Math.max(...shortTimeline.rounds.map((round, index) => Math.abs(round.end - shortExpected[index])));

  const hotTimeline = buildTimeline({
    anchorTime: 4,
    sourceIn,
    sourceOut,
    sourceBpm,
    ladder,
    countInBars: 1,
    roundCount: 20,
  });
  const prefix = hotTimeline.rounds.slice(0, 4);
  const hotLadder = ladder.map((bpm) => bpm + 6);
  const retimed = retimeFuture(hotTimeline, 4, hotLadder);
  const hotStartError = Math.abs(retimed.rounds[4].start - prefix[3].end);
  const hotTempoOk =
    retimed.rounds[4].bpm === hotLadder[4] && retimed.rounds[5].bpm === hotLadder[5];
  const prefixOk = prefix.every((round, index) => round.start === retimed.rounds[index].start);

  return [
    {
      id: "tenth",
      label: "第 10 轮边界",
      pass: near(tenthError, 0) && near(tenthPulseError, 0),
      value: `${tenth.start.toFixed(6)} s`,
      detail: `脉冲 ${tenth.pulseStart.toFixed(2)} · 误差 ${(tenthError * 1e6).toFixed(2)} µs`,
    },
    {
      id: "short",
      label: "32 次短循环",
      pass: near(shortError, 0),
      value: "1 beat × 32",
      detail: `最大边界误差 ${(shortError * 1e6).toFixed(2)} µs`,
    },
    {
      id: "hot",
      label: "播放中热改速",
      pass: near(hotStartError, 0) && hotTempoOk && prefixOk,
      value: `+6 BPM`,
      detail: `第 5 轮衔接误差 ${(hotStartError * 1e6).toFixed(2)} µs`,
    },
  ];
}
