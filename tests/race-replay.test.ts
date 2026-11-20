import { describe, expect, it } from "vitest";
import { ReviewEngine } from "../src/core/review-engine.ts";
import type {
  DraftFile,
  DraftStorage,
  EngineMode,
  ExportBundle,
  MediaIdentity,
  SpeechInterval,
} from "../src/core/types.ts";
import { ReplayHarness } from "../src/replay/replay-harness.ts";

const alpha: MediaIdentity = {
  id: "sha256:alpha",
  name: "alpha-long-take.mp4",
  sizeBytes: 8_388_608,
  lastModified: 1_730_000_000_000,
  fingerprint: "sha256:aaaaaaaaaaaaaaaa",
};

const beta: MediaIdentity = {
  id: "sha256:beta",
  name: "beta-long-take.mkv",
  sizeBytes: 12_582_912,
  lastModified: 1_730_000_100_000,
  fingerprint: "sha256:bbbbbbbbbbbbbbbb",
};

function createReplay(mode: EngineMode): ReplayHarness {
  let nextId = 0;
  let replay: ReplayHarness;
  const engine: ReviewEngine = new ReviewEngine({
    mode,
    now: (): number => replay.currentTimeMs,
    idFactory: () => `interval-${++nextId}`,
  });
  replay = new ReplayHarness({
    engine,
    startTimeMs: 0,
    defaultFrameDelayMs: 10,
  });
  return replay;
}

function intervalOf(exported: ExportBundle, sourceId: string): SpeechInterval[] {
  const file = exported.files.find((entry) => entry.media.id === sourceId);
  expect(file, `导出中应存在 ${sourceId}`).toBeDefined();
  return file?.intervals ?? [];
}

function expectSourceIdentity(bundle: ExportBundle, interval: SpeechInterval, media: MediaIdentity): void {
  const file = bundle.files.find((entry) => entry.media.id === interval.sourceId);
  expect(file?.media).toEqual(media);
  expect(interval.startAnchor.sourceId).toBe(media.id);
  expect(interval.endAnchor.sourceId).toBe(media.id);
}

function expectConsistentExport(bundle: ExportBundle): void {
  for (const file of bundle.files) {
    for (const interval of file.intervals) {
      expect(interval.sourceId).toBe(file.media.id);
      expect(interval.startAnchor.sourceId).toBe(file.media.id);
      expect(interval.endAnchor.sourceId).toBe(file.media.id);
      expect(interval.startAnchor.mediaTime).toBe(interval.start);
      expect(interval.endAnchor.mediaTime).toBe(interval.end);
    }
  }
}

class MemoryDraftStorage implements DraftStorage {
  draft: DraftFile | null = null;

  load(): unknown | null {
    return this.draft;
  }

  save(draft: DraftFile): void {
    this.draft = structuredClone(draft);
  }

  clear(): void {
    this.draft = null;
  }
}

describe("可控延时事件回放", () => {
  it("连续 keydown/keyup 只形成一个区间，重复事件不移动锚点", () => {
    const replay = createReplay("fixed");
    replay
      .switchSource(0, alpha)
      .keyDown(10)
      .requestFrame({ atMs: 20, mediaTime: 3, delayMs: 30 })
      .requestFrame({ atMs: 60, mediaTime: 3.5, delayMs: 30 })
      .keyDown(80)
      .keyUp(100)
      .keyUp(110)
      .run();

    const snapshot = replay.engine.getSnapshot();
    expect(snapshot.intervals).toHaveLength(1);
    expect(snapshot.intervals[0]).toMatchObject({
      sourceId: alpha.id,
      start: 3,
      end: 3.5,
      closeReason: "key-release",
    });
    expect(snapshot.activeRecording).toBeNull();
    expect(snapshot.journal.filter((entry) => entry.code === "key-repeat")).toHaveLength(2);

    const exported = replay.engine.exportBundle();
    expect(exported.intervalCount).toBe(1);
    expect(intervalOf(exported, alpha.id)).toHaveLength(1);
    expectSourceIdentity(exported, snapshot.intervals[0]!, alpha);
    expectConsistentExport(exported);
  });

  it("换源后的迟到帧在 fixed 中被票据拦截，在 legacy 中污染新源区间", () => {
    const fixed = createReplay("fixed");
    fixed
      .switchSource(0, alpha)
      .keyDown(10)
      .requestFrame({ atMs: 20, mediaTime: 10, delayMs: 20 })
      .requestFrame({ atMs: 50, mediaTime: 12, delayMs: 300 })
      .requestFrame({ atMs: 55, mediaTime: 12, delayMs: 10 })
      .switchSource(100, beta)
      .keyUp(110)
      .keyDown(120)
      .requestFrame({ atMs: 130, mediaTime: 1, delayMs: 20 })
      .requestFrame({ atMs: 140, mediaTime: 1.5, delayMs: 20 })
      .keyUp(400)
      .run();

    const fixedExport = fixed.engine.exportBundle();
    const fixedAlpha = intervalOf(fixedExport, alpha.id);
    const fixedBeta = intervalOf(fixedExport, beta.id);
    expect(fixedAlpha).toHaveLength(1);
    expect(fixedAlpha[0]).toMatchObject({
      start: 10,
      end: 12,
      closeReason: "source-change",
    });
    expect(fixedBeta).toHaveLength(1);
    expect(fixedBeta[0]).toMatchObject({ start: 1, end: 1.5, closeReason: "key-release" });
    expect(fixed.engine.getSnapshot().staleCallbacksIgnored).toBe(1);
    expectSourceIdentity(fixedExport, fixedBeta[0]!, beta);
    expectSourceIdentity(fixedExport, fixedAlpha[0]!, alpha);
    expectConsistentExport(fixedExport);

    const legacy = createReplay("legacy");
    legacy
      .switchSource(0, alpha)
      .keyDown(10)
      .requestFrame({ atMs: 20, mediaTime: 10, delayMs: 20 })
      .requestFrame({ atMs: 50, mediaTime: 12, delayMs: 300 })
      .requestFrame({ atMs: 55, mediaTime: 12, delayMs: 10 })
      .switchSource(100, beta)
      .keyUp(110)
      .keyDown(120)
      .requestFrame({ atMs: 130, mediaTime: 1, delayMs: 20 })
      .requestFrame({ atMs: 140, mediaTime: 1.5, delayMs: 20 })
      .keyUp(400)
      .run();

    const legacyBeta = intervalOf(legacy.engine.exportBundle(), beta.id);
    expect(legacyBeta).toHaveLength(1);
    expect(legacyBeta[0]).toMatchObject({ sourceId: beta.id, start: 1, end: 12 });
    expect(legacy.engine.getSnapshot().staleCallbacksIgnored).toBe(0);
  });

  it("快速跳转按最后实际呈现位置关闭，旧跳转代帧不会延长新区间", () => {
    const fixed = createReplay("fixed");
    fixed
      .switchSource(0, alpha)
      .keyDown(20)
      .requestFrame({ atMs: 25, mediaTime: 10, delayMs: 10 })
      .requestFrame({ atMs: 40, mediaTime: 12, delayMs: 10 })
      .requestFrame({ atMs: 60, mediaTime: 30, delayMs: 200 })
      .seek(80, 5)
      .keyUp(90)
      .keyDown(100)
      .requestFrame({ atMs: 110, mediaTime: 5.5, delayMs: 20 })
      .requestFrame({ atMs: 120, mediaTime: 6, delayMs: 20 })
      .keyUp(300)
      .run();

    const fixedIntervals = intervalOf(fixed.engine.exportBundle(), alpha.id);
    expect(fixedIntervals).toHaveLength(2);
    const fixedReverse = fixedIntervals.find((interval) => interval.closeReason === "reverse-seek");
    const fixedSecond = fixedIntervals.find((interval) => interval.closeReason === "key-release");
    expect(fixedReverse).toMatchObject({
      start: 10,
      end: 12,
      closeReason: "reverse-seek",
    });
    expect(fixedSecond).toMatchObject({
      start: 5.5,
      end: 6,
      closeReason: "key-release",
    });
    expect(fixed.engine.getSnapshot().staleCallbacksIgnored).toBe(1);
    expectConsistentExport(fixed.engine.exportBundle());

    const legacy = createReplay("legacy");
    legacy
      .switchSource(0, alpha)
      .keyDown(20)
      .requestFrame({ atMs: 25, mediaTime: 10, delayMs: 10 })
      .requestFrame({ atMs: 40, mediaTime: 12, delayMs: 10 })
      .requestFrame({ atMs: 60, mediaTime: 30, delayMs: 200 })
      .seek(80, 5)
      .keyUp(90)
      .keyDown(100)
      .requestFrame({ atMs: 110, mediaTime: 5.5, delayMs: 20 })
      .requestFrame({ atMs: 120, mediaTime: 6, delayMs: 20 })
      .keyUp(300)
      .run();

    const legacyIntervals = intervalOf(legacy.engine.exportBundle(), alpha.id);
    expect(legacyIntervals.find((interval) => interval.closeReason === "key-release")).toMatchObject({
      start: 5.5,
      end: 30,
    });
  });

  it("无 seek 事件的反向呈现也按前向最后位置关闭并闩锁到松键", () => {
    const replay = createReplay("fixed");
    replay
      .switchSource(0, alpha)
      .keyDown(10)
      .requestFrame({ atMs: 20, mediaTime: 5, delayMs: 10 })
      .requestFrame({ atMs: 30, mediaTime: 9, delayMs: 10 })
      .requestFrame({ atMs: 50, mediaTime: 3, delayMs: 10 })
      .keyUp(80)
      .requestFrame({ atMs: 90, mediaTime: 3.5, delayMs: 10 })
      .run();

    const snapshot = replay.engine.getSnapshot();
    expect(snapshot.intervals).toHaveLength(1);
    expect(snapshot.intervals[0]).toMatchObject({
      start: 5,
      end: 9,
      closeReason: "reverse-seek",
    });
    expect(snapshot.activeRecording).toBeNull();
    expect(snapshot.journal.some((entry) => entry.code === "reverse-presentation")).toBe(true);
  });

  it("关窗先落区间再存草稿，恢复后媒体身份和导出归属保持不变", () => {
    const storage = new MemoryDraftStorage();
    const replay = createReplay("fixed");
    replay
      .switchSource(0, alpha)
      .keyDown(10)
      .requestFrame({ atMs: 20, mediaTime: 20, delayMs: 10 })
      .requestFrame({ atMs: 30, mediaTime: 23, delayMs: 10 })
      .windowClose(50)
      .run();

    const draft = replay.lastClosedDraft;
    expect(draft).not.toBeNull();
    if (!draft) {
      throw new Error("关窗未产生草稿");
    }
    expect(draft.intervals).toHaveLength(1);
    expect(draft.intervals[0]).toMatchObject({
      sourceId: alpha.id,
      start: 20,
      end: 23,
      closeReason: "window-close",
    });
    storage.save(draft);

    let nextId = 0;
    const restored = new ReviewEngine({
      mode: "fixed",
      now: () => 999,
      idFactory: () => `restored-${++nextId}`,
    });
    restored.restoreDraft(storage.load());
    const exported = restored.exportBundle();
    expect(exported.files).toHaveLength(1);
    expect(exported.files[0]?.media).toEqual(alpha);
    expect(exported.files[0]?.intervals).toHaveLength(1);
    expectSourceIdentity(exported, exported.files[0]!.intervals[0]!, alpha);
    expectConsistentExport(exported);
    expect(restored.getSnapshot().activeRecording).toBeNull();
  });
});
