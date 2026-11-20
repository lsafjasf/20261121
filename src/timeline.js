export const MAX_ROUNDS = 128;
export const DEFAULT_LADDER = [70, 76, 82, 88, 94, 100, 106, 112];

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function normalizeLadder(values) {
  const ladder = values
    .map((value) => Math.round(Number(value)))
    .filter((value) => Number.isFinite(value) && value >= 20 && value <= 300);
  return ladder.length ? ladder : [90];
}

export function loopDuration(sourceDuration, sourceBpm, targetBpm) {
  return sourceDuration * (sourceBpm / targetBpm);
}

export function countInSeconds(countInBars, beatsPerBar, bpm) {
  return countInBars * beatsPerBar * (60 / bpm);
}

export function buildTimeline({
  anchorTime = 0,
  sourceIn,
  sourceOut,
  sourceBpm,
  ladder = DEFAULT_LADDER,
  countInBars = 1,
  beatsPerBar = 4,
  roundCount = MAX_ROUNDS,
}) {
  const normalizedLadder = normalizeLadder(ladder);
  const sourceDuration = sourceOut - sourceIn;
  if (!(sourceDuration > 0)) {
    throw new Error("Loop out point must be after the in point.");
  }
  if (!(sourceBpm > 0)) {
    throw new Error("Source tempo must be positive.");
  }

  const countInBeats = countInBars * beatsPerBar;
  const firstBpm = normalizedLadder[0];
  const loopStart = anchorTime + countInSeconds(countInBars, beatsPerBar, firstBpm);
  const loopBeats = sourceDuration * (sourceBpm / 60);
  const rounds = [];
  let cursor = loopStart;

  for (let index = 0; index < roundCount; index += 1) {
    const bpm = normalizedLadder[index % normalizedLadder.length];
    const duration = loopDuration(sourceDuration, sourceBpm, bpm);
    rounds.push({
      index,
      bpm,
      start: cursor,
      end: cursor + duration,
      duration,
      pulseStart: countInBeats + index * loopBeats,
    });
    cursor += duration;
  }

  return {
    anchorTime,
    sourceIn,
    sourceOut,
    sourceBpm,
    sourceDuration,
    ladder: normalizedLadder,
    countInBars,
    beatsPerBar,
    countInBeats,
    firstBpm,
    loopStart,
    loopBeats,
    rounds,
  };
}

export function retimeFuture(timeline, fromRoundIndex, nextLadder) {
  const firstKept = clamp(Math.floor(fromRoundIndex), 0, timeline.rounds.length);
  const ladder = normalizeLadder(nextLadder);
  const keptRounds = timeline.rounds.slice(0, firstKept).map((round) => ({ ...round }));
  let cursor = firstKept === 0 ? timeline.loopStart : keptRounds[firstKept - 1].end;

  for (let index = firstKept; index < timeline.rounds.length; index += 1) {
    const bpm = ladder[index % ladder.length];
    const duration = loopDuration(timeline.sourceDuration, timeline.sourceBpm, bpm);
    keptRounds.push({
      index,
      bpm,
      start: cursor,
      end: cursor + duration,
      duration,
      pulseStart: timeline.countInBeats + index * timeline.loopBeats,
    });
    cursor += duration;
  }

  return { ...timeline, ladder, rounds: keptRounds };
}

export function roundAtTime(timeline, time) {
  if (time < timeline.loopStart) return null;
  return (
    timeline.rounds.find((round) => time >= round.start && time < round.end) ??
    timeline.rounds.at(-1)
  );
}

export function nextRoundAt(timeline, time) {
  return timeline.rounds.find((round) => round.start > time) ?? null;
}

export function countInPulses(timeline) {
  const beat = 60 / timeline.firstBpm;
  return Array.from({ length: timeline.countInBeats }, (_, index) => ({
    index,
    time: timeline.anchorTime + index * beat,
    downbeat: index % timeline.beatsPerBar === 0,
    countIn: true,
  }));
}

export function roundPulses(timeline, roundIndex) {
  const round = timeline.rounds[roundIndex];
  if (!round) return [];
  const count = Math.max(1, Math.round(timeline.loopBeats));
  const beat = round.duration / timeline.loopBeats;
  return Array.from({ length: count }, (_, index) => ({
    index: round.pulseStart + index,
    time: round.start + index * beat,
    downbeat: index % timeline.beatsPerBar === 0,
    countIn: false,
    roundIndex,
  }));
}

export function formatClock(seconds, showMillis = true) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const millis = Math.floor((safe % 1) * 1000);
  const base = `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return showMillis ? `${base}.${String(millis).padStart(3, "0")}` : base;
}

export function formatPulse(pulse) {
  const safe = Math.max(0, pulse);
  const bar = Math.floor(safe / 4) + 1;
  const beat = Math.floor(safe % 4) + 1;
  const tick = Math.floor((safe % 1) * 100);
  return `${String(bar).padStart(3, "0")}:${beat}:${String(tick).padStart(2, "0")}`;
}
