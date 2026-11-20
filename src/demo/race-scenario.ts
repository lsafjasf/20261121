import { ReviewEngine } from "../core/review-engine.ts";
import type { EngineMode, ExportBundle, MediaIdentity, SpeechInterval } from "../core/types.ts";
import { formatInterval, ReplayHarness } from "../replay/replay-harness.ts";

export const demoAlpha: MediaIdentity = {
  id: "sha256:demo-alpha",
  name: "访谈-A-长镜头.mp4",
  sizeBytes: 734_003_200,
  lastModified: 1_730_000_000_000,
  fingerprint: "sha256-sampled:demo-alpha:524288+524288",
};

export const demoBeta: MediaIdentity = {
  id: "sha256:demo-beta",
  name: "访谈-B-补拍.mkv",
  sizeBytes: 1_048_576_000,
  lastModified: 1_730_000_100_000,
  fingerprint: "sha256-sampled:demo-beta:524288+524288",
};

export interface SideResult {
  mode: EngineMode;
  bundle: ExportBundle;
  betaIntervals: SpeechInterval[];
  staleCallbacksIgnored: number;
}

export interface RaceComparison {
  fixed: SideResult;
  legacy: SideResult;
}

function runSide(mode: EngineMode): SideResult {
  let nextId = 0;
  let replay: ReplayHarness;
  const engine: ReviewEngine = new ReviewEngine({
    mode,
    now: (): number => replay.currentTimeMs,
    idFactory: () => `${mode}-${++nextId}`,
  });
  replay = new ReplayHarness({ engine, startTimeMs: 0, defaultFrameDelayMs: 20 });
  replay
    .switchSource(0, demoAlpha)
    .keyDown(10)
    .requestFrame({ atMs: 20, mediaTime: 10, delayMs: 20 })
    .requestFrame({ atMs: 50, mediaTime: 12, delayMs: 300 })
    .requestFrame({ atMs: 55, mediaTime: 12, delayMs: 10 })
    .switchSource(100, demoBeta)
    .keyUp(110)
    .keyDown(120)
    .requestFrame({ atMs: 130, mediaTime: 1, delayMs: 20 })
    .requestFrame({ atMs: 140, mediaTime: 1.5, delayMs: 20 })
    .keyUp(400)
    .run();

  const bundle = engine.exportBundle();
  const betaIntervals =
    bundle.files.find((file) => file.media.id === demoBeta.id)?.intervals ?? [];
  return {
    mode,
    bundle,
    betaIntervals,
    staleCallbacksIgnored: engine.getSnapshot().staleCallbacksIgnored,
  };
}

export function runDelayedSwitchRace(): RaceComparison {
  return {
    fixed: runSide("fixed"),
    legacy: runSide("legacy"),
  };
}

export function summarizeRace(comparison: RaceComparison): string[] {
  const fixed = comparison.fixed.betaIntervals[0];
  const legacy = comparison.legacy.betaIntervals[0];
  return [
    `修复版：${fixed ? formatInterval(fixed) : "无新区间"}；拦截迟到帧 ${comparison.fixed.staleCallbacksIgnored} 次`,
    `旧逻辑：${legacy ? formatInterval(legacy) : "无新区间"}；拦截迟到帧 ${comparison.legacy.staleCallbacksIgnored} 次`,
  ];
}
