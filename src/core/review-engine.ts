import type {
  ActiveRecording,
  CloseReason,
  DraftFile,
  EngineMode,
  EngineSnapshot,
  ExportBundle,
  FrameResult,
  FrameTicket,
  JournalCode,
  JournalEntry,
  KeyResult,
  MediaIdentity,
  PresentedAnchor,
  PresentedFrame,
  SeekResult,
  SpeechInterval,
} from "./types.ts";

export interface ReviewEngineOptions {
  mode?: EngineMode;
  now?: () => number;
  idFactory?: () => string;
  minimumIntervalSeconds?: number;
  reverseEpsilonSeconds?: number;
}

type SnapshotListener = (snapshot: EngineSnapshot) => void;

const cloneIdentity = (media: MediaIdentity): MediaIdentity => ({ ...media });
const cloneAnchor = (anchor: PresentedAnchor): PresentedAnchor => ({ ...anchor });
const cloneInterval = (interval: SpeechInterval): SpeechInterval => ({
  ...interval,
  startAnchor: cloneAnchor(interval.startAnchor),
  endAnchor: cloneAnchor(interval.endAnchor),
});

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

export class ReviewEngine {
  readonly sessionId: string;
  readonly mode: EngineMode;

  private readonly now: () => number;
  private readonly idFactory: () => string;
  private readonly minimumIntervalSeconds: number;
  private readonly reverseEpsilonSeconds: number;
  private readonly listeners = new Set<SnapshotListener>();

  private sources = new Map<string, MediaIdentity>();
  private intervals: SpeechInterval[] = [];
  private journal: JournalEntry[] = [];

  private activeSourceId: string | null = null;
  private sourceGeneration = 0;
  private seekGeneration = 0;
  private requestSequence = 0;
  private latestPresented: PresentedAnchor | null = null;
  private activeRecording: ActiveRecording | null = null;

  private held = false;
  private latched = false;
  private armed = false;
  private staleCallbacksIgnored = 0;

  constructor(options: ReviewEngineOptions = {}) {
    this.mode = options.mode ?? "fixed";
    this.now = options.now ?? (() => Date.now());
    this.idFactory = options.idFactory ?? (() => crypto.randomUUID());
    this.minimumIntervalSeconds = options.minimumIntervalSeconds ?? 0;
    this.reverseEpsilonSeconds = options.reverseEpsilonSeconds ?? 0.001;
    this.sessionId = crypto.randomUUID();
  }

  subscribe(listener: SnapshotListener): () => void {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): EngineSnapshot {
    return {
      mode: this.mode,
      held: this.held,
      latched: this.latched,
      armed: this.armed,
      activeSource: this.getActiveSource(),
      latestPresented: this.latestPresented ? cloneAnchor(this.latestPresented) : null,
      activeRecording: this.activeRecording
        ? {
            ...this.activeRecording,
            startAnchor: cloneAnchor(this.activeRecording.startAnchor),
            lastForwardAnchor: cloneAnchor(this.activeRecording.lastForwardAnchor),
          }
        : null,
      sources: [...this.sources.values()].map(cloneIdentity),
      intervals: this.intervals.map(cloneInterval),
      journal: this.journal.map((entry) => ({ ...entry })),
      staleCallbacksIgnored: this.staleCallbacksIgnored,
    };
  }

  activateSource(media: MediaIdentity): void {
    if (!this.isValidIdentity(media)) {
      throw new Error("无效的媒体身份");
    }

    const previousSourceId = this.activeSourceId;
    this.sources.set(media.id, cloneIdentity(media));

    if (previousSourceId === media.id) {
      this.emit();
      return;
    }

    if (this.activeRecording) {
      this.finishActive("source-change");
    }

    if (this.held) {
      this.armed = false;
      this.latched = true;
    }

    this.activeSourceId = media.id;
    this.sourceGeneration += 1;
    this.seekGeneration = 0;
    this.latestPresented = null;
    this.requestSequence = 0;
    this.emit();
  }

  clearActiveSource(): void {
    if (this.activeRecording) {
      this.finishActive("manual");
    }
    this.activeSourceId = null;
    this.latestPresented = null;
    this.armed = false;
    this.latched = false;
    this.held = false;
    this.emit();
  }

  issueFrameTicket(): FrameTicket | null {
    const source = this.getActiveSource();
    if (!source) {
      return null;
    }

    this.requestSequence += 1;
    return {
      sourceId: source.id,
      sourceGeneration: this.sourceGeneration,
      seekGeneration: this.seekGeneration,
      requestId: this.requestSequence,
      requestedAt: this.now(),
    };
  }

  onFramePresented(frame: PresentedFrame): FrameResult {
    const source = this.getActiveSource();
    if (!source) {
      return { status: "ignored-no-source", started: false, closed: null };
    }

    if (this.mode === "fixed" && !this.isCurrentTicket(frame.ticket)) {
      this.staleCallbacksIgnored += 1;
      this.addJournal(
        "stale-frame-ignored",
        `忽略迟到帧：ticket ${frame.ticket.sourceId}/源代${frame.ticket.sourceGeneration}/跳转代${frame.ticket.seekGeneration}/请求${frame.ticket.requestId}`,
        frame.ticket.sourceId,
      );
      this.emit();
      return { status: "ignored-stale", started: false, closed: null };
    }

    if (!isFiniteNumber(frame.mediaTime) || frame.mediaTime < 0) {
      this.staleCallbacksIgnored += 1;
      this.emit();
      return { status: "ignored-stale", started: false, closed: null };
    }

    const anchor: PresentedAnchor = {
      sourceId: source.id,
      sourceGeneration: this.sourceGeneration,
      seekGeneration: this.seekGeneration,
      requestId: frame.ticket.requestId,
      mediaTime: frame.mediaTime,
      presentedFrames: frame.presentedFrames,
      callbackAt: frame.callbackAt,
    };

    const previous = this.latestPresented;
    const movedBackward =
      previous !== null &&
      anchor.mediaTime + this.reverseEpsilonSeconds < previous.mediaTime;

    let closed: SpeechInterval | null = null;
    if (movedBackward) {
      this.addJournal(
        "reverse-presentation",
        `实际呈现位置反向：${previous?.mediaTime.toFixed(3)}s → ${anchor.mediaTime.toFixed(3)}s`,
        source.id,
      );
      if (this.activeRecording) {
        closed = this.finishActive("reverse-seek");
      }
      if (this.held) {
        this.latched = true;
      }
      this.armed = false;
    }

    this.latestPresented = anchor;
    let started = false;

    if (this.activeRecording) {
      if (anchor.mediaTime < this.activeRecording.lastForwardAnchor.mediaTime) {
        closed = this.finishActive("reverse-seek") ?? closed;
        if (this.held) {
          this.latched = true;
        }
      } else {
        this.activeRecording.lastForwardAnchor = anchor;
      }
    } else if (this.armed && this.held && !this.latched) {
      this.startActive(anchor);
      started = true;
    }

    this.emit();
    return { status: "accepted", started, closed };
  }

  requestSeek(targetTime: number): SeekResult {
    const source = this.getActiveSource();
    if (!source) {
      return { closed: null, closeReason: null };
    }

    const previousAnchor = this.activeRecording?.lastForwardAnchor ?? this.latestPresented;
    const reason: CloseReason =
      previousAnchor &&
      isFiniteNumber(targetTime) &&
      targetTime + this.reverseEpsilonSeconds < previousAnchor.mediaTime
        ? "reverse-seek"
        : "forward-seek";

    this.seekGeneration += 1;
    this.latestPresented = null;
    this.armed = false;

    let closed: SpeechInterval | null = null;
    if (this.activeRecording) {
      closed = this.finishActive(reason);
    }
    if (this.held) {
      this.latched = true;
    }

    this.emit();
    return { closed, closeReason: closed ? closed.closeReason : null };
  }

  handleKeyDown(): KeyResult {
    if (this.held) {
      this.addJournal("key-repeat", "按住键的重复 keydown 已忽略");
      this.emit();
      return { status: "repeat-ignored" };
    }

    this.held = true;
    this.latched = false;

    if (!this.activeSourceId) {
      this.addJournal("key-without-source", "没有活动媒体，无法开始区间");
      this.emit();
      return { status: "no-active-source" };
    }

    if (this.latestPresented) {
      this.startActive(this.latestPresented);
      this.emit();
      return { status: "opened" };
    }

    this.armed = true;
    this.addJournal("recording-armed", "等待下一帧已呈现画面后开始区间");
    this.emit();
    return { status: "armed" };
  }

  handleKeyUp(): KeyResult {
    if (!this.held) {
      this.addJournal("key-repeat", "多余 keyup 已忽略");
      this.emit();
      return { status: "repeat-ignored" };
    }

    this.held = false;
    this.latched = false;
    this.armed = false;

    if (!this.activeRecording) {
      this.emit();
      return { status: "latched" };
    }

    const closed = this.finishActive("key-release");
    this.emit();
    return closed ? { status: "closed", interval: cloneInterval(closed) } : { status: "closed" };
  }

  handleWindowClose(): DraftFile {
    if (this.activeRecording) {
      const closed = this.finishActive("window-close");
      if (closed) {
        this.addJournal(
          "window-close-closed-interval",
          `关窗前按最后呈现位置 ${closed.end.toFixed(3)}s 关闭区间`,
          closed.sourceId,
        );
      }
    }
    this.held = false;
    this.armed = false;
    this.latched = false;
    this.latestPresented = null;
    const draft = this.serializeDraft();
    this.emit();
    return draft;
  }

  serializeDraft(): DraftFile {
    return {
      format: "speech-review-draft",
      version: 1,
      savedAt: this.now(),
      sources: [...this.sources.values()].map(cloneIdentity),
      intervals: this.intervals.map(cloneInterval),
      journal: this.journal.map((entry) => ({ ...entry })),
    };
  }

  restoreDraft(value: unknown): void {
    const draft = this.parseDraft(value);
    this.sources = new Map(draft.sources.map((media) => [media.id, cloneIdentity(media)]));
    this.intervals = draft.intervals.map(cloneInterval);
    this.journal = draft.journal.map((entry) => ({ ...entry }));
    this.activeSourceId = null;
    this.activeRecording = null;
    this.latestPresented = null;
    this.held = false;
    this.latched = false;
    this.armed = false;
    this.staleCallbacksIgnored = 0;
    this.sourceGeneration += 1;
    this.seekGeneration = 0;
    this.requestSequence = 0;
    this.emit();
  }

  exportBundle(): ExportBundle {
    const files = [...this.sources.values()]
      .map((media) => ({
        media: cloneIdentity(media),
        intervals: this.intervals
          .filter((interval) => interval.sourceId === media.id)
          .map(cloneInterval)
          .sort((left, right) => left.start - right.start || left.end - right.end),
      }))
      .sort((left, right) => left.media.id.localeCompare(right.media.id));

    return {
      format: "speech-review-export",
      version: 1,
      sessionId: this.sessionId,
      exportedAt: this.now(),
      mode: this.mode,
      files,
      intervalCount: this.intervals.length,
      sourceCount: this.sources.size,
    };
  }

  private startActive(anchor: PresentedAnchor): void {
    this.armed = false;
    this.activeRecording = {
      intervalId: this.idFactory(),
      sourceId: anchor.sourceId,
      openedAt: this.now(),
      startAnchor: cloneAnchor(anchor),
      lastForwardAnchor: cloneAnchor(anchor),
    };
    this.addJournal(
      "recording-started",
      `从实际呈现位置 ${anchor.mediaTime.toFixed(3)}s 开始区间`,
      anchor.sourceId,
    );
  }

  private finishActive(reason: CloseReason): SpeechInterval | null {
    const active = this.activeRecording;
    if (!active) {
      return null;
    }

    this.activeRecording = null;
    const endAnchor = active.lastForwardAnchor;
    const start = active.startAnchor.mediaTime;
    const end = endAnchor.mediaTime;
    const duration = end - start;

    if (duration <= this.minimumIntervalSeconds) {
      this.addJournal(
        "interval-discarded",
        `区间未形成有效时长（${start.toFixed(3)}s → ${end.toFixed(3)}s，${reason}）`,
        active.sourceId,
      );
      return null;
    }

    const interval: SpeechInterval = {
      id: active.intervalId,
      sourceId: active.sourceId,
      start,
      end,
      duration,
      closeReason: reason,
      openedAt: active.openedAt,
      closedAt: this.now(),
      startAnchor: cloneAnchor(active.startAnchor),
      endAnchor: cloneAnchor(endAnchor),
    };
    this.intervals.push(interval);
    this.addJournal(
      "interval-committed",
      `提交区间 ${start.toFixed(3)}s–${end.toFixed(3)}s（${reason}）`,
      active.sourceId,
    );
    if (reason === "source-change") {
      this.addJournal(
        "source-change-closed-interval",
        "换源时先按旧媒体最后呈现位置关闭区间",
        active.sourceId,
      );
    }
    return cloneInterval(interval);
  }

  private getActiveSource(): MediaIdentity | null {
    if (!this.activeSourceId) {
      return null;
    }
    const source = this.sources.get(this.activeSourceId);
    return source ? cloneIdentity(source) : null;
  }

  private isCurrentTicket(ticket: FrameTicket): boolean {
    const source = this.getActiveSource();
    return (
      source !== null &&
      ticket.sourceId === source.id &&
      ticket.sourceGeneration === this.sourceGeneration &&
      ticket.seekGeneration === this.seekGeneration
    );
  }

  private isValidIdentity(value: unknown): value is MediaIdentity {
    if (typeof value !== "object" || value === null) {
      return false;
    }
    const candidate = value as Record<string, unknown>;
    return (
      typeof candidate.id === "string" &&
      candidate.id.length > 0 &&
      typeof candidate.name === "string" &&
      typeof candidate.sizeBytes === "number" &&
      Number.isFinite(candidate.sizeBytes) &&
      typeof candidate.lastModified === "number" &&
      Number.isFinite(candidate.lastModified) &&
      typeof candidate.fingerprint === "string"
    );
  }

  private parseDraft(value: unknown): DraftFile {
    if (typeof value !== "object" || value === null) {
      throw new Error("草稿不是对象");
    }
    const candidate = value as Record<string, unknown>;
    if (
      candidate.format !== "speech-review-draft" ||
      candidate.version !== 1 ||
      !Array.isArray(candidate.sources) ||
      !Array.isArray(candidate.intervals)
    ) {
      throw new Error("草稿格式或版本不受支持");
    }

    if (!candidate.sources.every((source) => this.isValidIdentity(source))) {
      throw new Error("草稿包含无效媒体身份");
    }
    const sources = candidate.sources.map((source) => cloneIdentity(source));
    const sourceIds = new Set(sources.map((source) => source.id));
    if (sourceIds.size !== sources.length) {
      throw new Error("草稿包含重复媒体身份");
    }

    const intervalIds = new Set<string>();
    const intervals: SpeechInterval[] = [];
    for (const value of candidate.intervals) {
      const interval = this.parseInterval(value, sourceIds);
      if (intervalIds.has(interval.id)) {
        throw new Error(`草稿包含重复区间 ${interval.id}`);
      }
      intervalIds.add(interval.id);
      intervals.push(interval);
    }

    const journal = Array.isArray(candidate.journal)
      ? candidate.journal
          .filter((entry): entry is JournalEntry => this.isJournalEntry(entry))
          .map((entry) => ({ ...entry }))
      : [];

    return {
      format: "speech-review-draft",
      version: 1,
      savedAt: isFiniteNumber(candidate.savedAt) ? candidate.savedAt : this.now(),
      sources,
      intervals,
      journal,
    };
  }

  private parseInterval(value: unknown, sourceIds: Set<string>): SpeechInterval {
    if (typeof value !== "object" || value === null) {
      throw new Error("草稿包含无效区间");
    }
    const candidate = value as Record<string, unknown>;
    if (
      typeof candidate.id !== "string" ||
      typeof candidate.sourceId !== "string" ||
      !sourceIds.has(candidate.sourceId) ||
      !isFiniteNumber(candidate.start) ||
      !isFiniteNumber(candidate.end) ||
      candidate.end <= candidate.start ||
      !this.isCloseReason(candidate.closeReason)
    ) {
      throw new Error("草稿包含损坏的区间");
    }
    const startAnchor = this.parseAnchor(candidate.startAnchor, candidate.sourceId);
    const endAnchor = this.parseAnchor(candidate.endAnchor, candidate.sourceId);

    return {
      id: candidate.id,
      sourceId: candidate.sourceId,
      start: candidate.start,
      end: candidate.end,
      duration: candidate.end - candidate.start,
      closeReason: candidate.closeReason,
      openedAt: isFiniteNumber(candidate.openedAt) ? candidate.openedAt : 0,
      closedAt: isFiniteNumber(candidate.closedAt) ? candidate.closedAt : 0,
      startAnchor,
      endAnchor,
    };
  }

  private parseAnchor(value: unknown, sourceId: string): PresentedAnchor {
    if (typeof value !== "object" || value === null) {
      throw new Error("草稿缺少实际呈现锚点");
    }
    const candidate = value as Record<string, unknown>;
    if (
      candidate.sourceId !== sourceId ||
      !isFiniteNumber(candidate.sourceGeneration) ||
      !isFiniteNumber(candidate.seekGeneration) ||
      !isFiniteNumber(candidate.requestId) ||
      !isFiniteNumber(candidate.mediaTime) ||
      !isFiniteNumber(candidate.presentedFrames) ||
      !isFiniteNumber(candidate.callbackAt)
    ) {
      throw new Error("草稿包含无效呈现锚点");
    }
    return {
      sourceId,
      sourceGeneration: candidate.sourceGeneration,
      seekGeneration: candidate.seekGeneration,
      requestId: candidate.requestId,
      mediaTime: candidate.mediaTime,
      presentedFrames: candidate.presentedFrames,
      callbackAt: candidate.callbackAt,
    };
  }

  private isJournalEntry(value: unknown): value is JournalEntry {
    if (typeof value !== "object" || value === null) {
      return false;
    }
    const candidate = value as Record<string, unknown>;
    return (
      isFiniteNumber(candidate.at) &&
      typeof candidate.code === "string" &&
      typeof candidate.message === "string" &&
      (candidate.sourceId === undefined || typeof candidate.sourceId === "string")
    );
  }

  private isCloseReason(value: unknown): value is CloseReason {
    return (
      value === "key-release" ||
      value === "forward-seek" ||
      value === "reverse-seek" ||
      value === "source-change" ||
      value === "window-close" ||
      value === "manual"
    );
  }

  private addJournal(code: JournalCode, message: string, sourceId?: string): void {
    const entry: JournalEntry =
      sourceId === undefined
        ? { at: this.now(), code, message }
        : { at: this.now(), code, message, sourceId };
    this.journal.push(entry);
    if (this.journal.length > 300) {
      this.journal.splice(0, this.journal.length - 300);
    }
  }

  private emit(): void {
    const snapshot = this.getSnapshot();
    for (const listener of this.listeners) {
      listener(snapshot);
    }
  }
}
