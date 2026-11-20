import { ReviewEngine } from "../core/review-engine.ts";
import type {
  DraftFile,
  FrameResult,
  MediaIdentity,
  PresentedFrame,
  SpeechInterval,
} from "../core/types.ts";

export interface ReplayHarnessOptions {
  engine: ReviewEngine;
  defaultFrameDelayMs?: number;
  startTimeMs?: number;
}

export interface FrameRequest {
  atMs: number;
  mediaTime: number;
  presentedFrames?: number;
  delayMs?: number;
}

export interface ReplayTraceEntry {
  atMs: number;
  kind: string;
  detail: string;
}

interface ScheduledTask {
  atMs: number;
  sequence: number;
  label: string;
  run: () => void;
}

export class ReplayHarness {
  readonly engine: ReviewEngine;
  readonly trace: ReplayTraceEntry[] = [];

  private nowMs: number;
  private nextSequence = 0;
  private readonly queue: ScheduledTask[] = [];
  private readonly defaultFrameDelayMs: number;

  constructor(options: ReplayHarnessOptions) {
    this.engine = options.engine;
    this.defaultFrameDelayMs = options.defaultFrameDelayMs ?? 80;
    this.nowMs = options.startTimeMs ?? 1_000;
  }

  get currentTimeMs(): number {
    return this.nowMs;
  }

  switchSource(atMs: number, media: MediaIdentity): this {
    return this.schedule(atMs, "switch-source", `切换到 ${media.name}`, () => {
      this.engine.activateSource(media);
    });
  }

  keyDown(atMs: number): this {
    return this.schedule(atMs, "key-down", "按下记录键", () => {
      this.engine.handleKeyDown();
    });
  }

  keyUp(atMs: number): this {
    return this.schedule(atMs, "key-up", "松开记录键", () => {
      this.engine.handleKeyUp();
    });
  }

  seek(atMs: number, targetSeconds: number, label = "用户跳转"): this {
    return this.schedule(atMs, "seek", `${label} → ${targetSeconds.toFixed(3)}s`, () => {
      this.engine.requestSeek(targetSeconds);
    });
  }

  requestFrame(request: FrameRequest): this {
    return this.schedule(request.atMs, "request-frame", `请求 ${request.mediaTime.toFixed(3)}s 的帧`, () => {
      const ticket = this.engine.issueFrameTicket();
      if (!ticket) {
        this.record(request.atMs, "frame-skipped", "没有活动媒体");
        return;
      }
      const delayMs = request.delayMs ?? this.defaultFrameDelayMs;
      const deliveredAt = request.atMs + delayMs;
      this.schedule(deliveredAt, "deliver-frame", `投递 ${request.mediaTime.toFixed(3)}s 的帧`, () => {
        const frame: PresentedFrame = {
          ticket,
          mediaTime: request.mediaTime,
          presentedFrames: request.presentedFrames ?? 1,
          callbackAt: this.nowMs,
        };
        const result = this.engine.onFramePresented(frame);
        this.describeFrameResult(deliveredAt, result);
      });
    });
  }

  windowClose(atMs: number): this {
    return this.schedule(atMs, "window-close", "关闭窗口", () => {
      const draft = this.engine.handleWindowClose();
      this.closedDraft = draft;
    });
  }

  schedule(atMs: number, kind: string, detail: string, action: () => void): this {
    if (!Number.isFinite(atMs) || atMs < this.nowMs) {
      throw new Error(`事件时间 ${atMs} 早于当前时钟 ${this.nowMs}`);
    }
    this.nextSequence += 1;
    this.queue.push({
      atMs,
      sequence: this.nextSequence,
      label: kind,
      run: () => {
        action();
        this.record(atMs, kind, detail);
      },
    });
    return this;
  }

  run(): this {
    while (this.queue.length > 0) {
      this.queue.sort((left, right) => left.atMs - right.atMs || left.sequence - right.sequence);
      const task = this.queue.shift();
      if (!task) {
        break;
      }
      if (task.atMs < this.nowMs) {
        throw new Error(`回放时钟倒退：${this.nowMs} → ${task.atMs}`);
      }
      this.nowMs = task.atMs;
      task.run();
    }
    return this;
  }

  get lastClosedDraft(): DraftFile | null {
    return this.closedDraft;
  }

  private closedDraft: DraftFile | null = null;

  private record(atMs: number, kind: string, detail: string): void {
    this.trace.push({ atMs, kind, detail });
  }

  private describeFrameResult(atMs: number, result: FrameResult): void {
    const closed = result.closed ? formatInterval(result.closed) : "";
    const detail = [
      result.status,
      result.started ? "started" : "",
      closed ? `closed ${closed}` : "",
    ]
      .filter(Boolean)
      .join(" ");
    this.record(atMs, "frame-result", detail);
  }
}

export const formatInterval = (interval: SpeechInterval): string =>
  `[${interval.start.toFixed(3)}, ${interval.end.toFixed(3)}) ${interval.closeReason} @${interval.sourceId}`;
