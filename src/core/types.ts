export type EngineMode = "fixed" | "legacy";

export type CloseReason =
  | "key-release"
  | "forward-seek"
  | "reverse-seek"
  | "source-change"
  | "window-close"
  | "manual";

export interface MediaIdentity {
  id: string;
  name: string;
  sizeBytes: number;
  lastModified: number;
  fingerprint: string;
}

export interface PresentedAnchor {
  sourceId: string;
  sourceGeneration: number;
  seekGeneration: number;
  requestId: number;
  mediaTime: number;
  presentedFrames: number;
  callbackAt: number;
}

export interface FrameTicket {
  sourceId: string;
  sourceGeneration: number;
  seekGeneration: number;
  requestId: number;
  requestedAt: number;
}

export interface PresentedFrame {
  ticket: FrameTicket;
  mediaTime: number;
  presentedFrames: number;
  callbackAt: number;
}

export interface SpeechInterval {
  id: string;
  sourceId: string;
  start: number;
  end: number;
  duration: number;
  closeReason: CloseReason;
  openedAt: number;
  closedAt: number;
  startAnchor: PresentedAnchor;
  endAnchor: PresentedAnchor;
}

export interface ActiveRecording {
  intervalId: string;
  sourceId: string;
  openedAt: number;
  startAnchor: PresentedAnchor;
  lastForwardAnchor: PresentedAnchor;
}

export type JournalCode =
  | "key-repeat"
  | "key-without-source"
  | "recording-armed"
  | "recording-started"
  | "interval-committed"
  | "interval-discarded"
  | "reverse-presentation"
  | "stale-frame-ignored"
  | "source-change-closed-interval"
  | "window-close-closed-interval";

export interface JournalEntry {
  at: number;
  code: JournalCode;
  message: string;
  sourceId?: string;
}

export interface ExportedFile {
  media: MediaIdentity;
  intervals: SpeechInterval[];
}

export interface ExportBundle {
  format: "speech-review-export";
  version: 1;
  sessionId: string;
  exportedAt: number;
  mode: EngineMode;
  files: ExportedFile[];
  intervalCount: number;
  sourceCount: number;
}

export interface DraftFile {
  format: "speech-review-draft";
  version: 1;
  savedAt: number;
  sources: MediaIdentity[];
  intervals: SpeechInterval[];
  journal: JournalEntry[];
}

export interface FrameResult {
  status: "accepted" | "ignored-stale" | "ignored-no-source";
  started: boolean;
  closed: SpeechInterval | null;
}

export interface SeekResult {
  closed: SpeechInterval | null;
  closeReason: CloseReason | null;
}

export interface KeyResult {
  status:
    | "opened"
    | "armed"
    | "closed"
    | "repeat-ignored"
    | "no-active-source"
    | "latched";
  interval?: SpeechInterval;
}

export interface EngineSnapshot {
  mode: EngineMode;
  held: boolean;
  latched: boolean;
  armed: boolean;
  activeSource: MediaIdentity | null;
  latestPresented: PresentedAnchor | null;
  activeRecording: ActiveRecording | null;
  sources: MediaIdentity[];
  intervals: SpeechInterval[];
  journal: JournalEntry[];
  staleCallbacksIgnored: number;
}

export interface DraftStorage {
  load(): unknown | null;
  save(draft: DraftFile): void;
  clear(): void;
}
