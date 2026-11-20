import { ReviewEngine } from "../core/review-engine.ts";
import type { PresentedFrame } from "../core/types.ts";

interface FrameMetadataLike {
  mediaTime: number;
  presentedFrames: number;
}

export class VideoFrameBridge {
  private readonly video: HTMLVideoElement;
  private readonly engine: ReviewEngine;
  private readonly supportsFrameCallback: boolean;
  private running = false;
  private handle: number | null = null;
  private localGeneration = 0;

  constructor(video: HTMLVideoElement, engine: ReviewEngine) {
    this.video = video;
    this.engine = engine;
    this.supportsFrameCallback = "requestVideoFrameCallback" in video;
  }

  get usesActualPresentation(): boolean {
    return this.supportsFrameCallback;
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    this.requestNext();
  }

  stop(): void {
    this.running = false;
    this.cancelPending();
    this.localGeneration += 1;
  }

  invalidate(): void {
    this.localGeneration += 1;
    this.cancelPending();
    if (this.running) {
      this.requestNext();
    }
  }

  private requestNext(): void {
    if (!this.running || this.handle !== null) {
      return;
    }
    const ticket = this.engine.issueFrameTicket();
    if (!ticket) {
      return;
    }
    const generation = this.localGeneration;

    if (this.supportsFrameCallback) {
      this.handle = this.video.requestVideoFrameCallback((_now, metadata) => {
        this.handle = null;
        if (!this.running || generation !== this.localGeneration) {
          this.requestNext();
          return;
        }
        this.deliver(ticket, {
          mediaTime: metadata.mediaTime,
          presentedFrames: metadata.presentedFrames,
        });
        this.requestNext();
      });
      return;
    }

    this.handle = requestAnimationFrame(() => {
      this.handle = null;
      if (!this.running || generation !== this.localGeneration) {
        this.requestNext();
        return;
      }
      this.deliver(ticket, {
        mediaTime: this.video.currentTime,
        presentedFrames: 1,
      });
      this.requestNext();
    });
  }

  private deliver(ticket: PresentedFrame["ticket"], metadata: FrameMetadataLike): void {
    const frame: PresentedFrame = {
      ticket,
      mediaTime: metadata.mediaTime,
      presentedFrames: metadata.presentedFrames,
      callbackAt: Date.now(),
    };
    this.engine.onFramePresented(frame);
  }

  private cancelPending(): void {
    if (this.handle === null) {
      return;
    }
    if (this.supportsFrameCallback) {
      this.video.cancelVideoFrameCallback(this.handle);
    } else {
      cancelAnimationFrame(this.handle);
    }
    this.handle = null;
  }
}
