import "./styles.css";
import "@fontsource-variable/noto-sans-sc";
import "@fontsource/dm-mono/400.css";
import "@fontsource/dm-mono/500.css";
import { LocalStorageDraftStore, MemoryDraftStore } from "./core/draft-storage.ts";
import { ReviewEngine } from "./core/review-engine.ts";
import type {
  DraftStorage,
  EngineMode,
  EngineSnapshot,
  MediaIdentity,
  SpeechInterval,
} from "./core/types.ts";
import { runDelayedSwitchRace } from "./demo/race-scenario.ts";
import { computeMediaIdentity } from "./media/file-identity.ts";
import { VideoFrameBridge } from "./media/video-frame-bridge.ts";
import { appTemplate } from "./ui/template.ts";

const root = document.querySelector<HTMLDivElement>("#app");
if (!root) {
  throw new Error("缺少 #app 挂载点");
}
root.innerHTML = appTemplate;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`缺少界面元素 ${selector}`);
  }
  return element;
}

const modeSelect = required<HTMLSelectElement>("#mode-select");
const modeLabel = required<HTMLElement>("#session-mode-label");
const legacyBanner = required<HTMLElement>("#legacy-banner");
const activeSourceName = required<HTMLElement>("#active-source-name");
const activeSourceId = required<HTMLElement>("#active-source-id");
const videoShell = required<HTMLElement>("#video-shell");
const video = required<HTMLVideoElement>("#video");
const presentationBadge = required<HTMLElement>("#presentation-badge");
const recordingHud = required<HTMLElement>("#recording-hud");
const recordingHudLabel = required<HTMLElement>("#recording-hud-label");
const playButton = required<HTMLButtonElement>("#play-button");
const backButton = required<HTMLButtonElement>("#back-button");
const forwardButton = required<HTMLButtonElement>("#forward-button");
const muteButton = required<HTMLButtonElement>("#mute-button");
const timeline = required<HTMLInputElement>("#timeline");
const currentTimeLabel = required<HTMLElement>("#current-time");
const durationTimeLabel = required<HTMLElement>("#duration-time");
const recordStatus = required<HTMLElement>("#record-status");
const staleCount = required<HTMLElement>("#stale-count");
const intervalCount = required<HTMLElement>("#interval-count");
const panelIntervalCount = required<HTMLElement>("#panel-interval-count");
const sourceList = required<HTMLElement>("#source-list");
const intervalList = required<HTMLElement>("#interval-list");
const journalList = required<HTMLElement>("#journal-list");
const fileInput = required<HTMLInputElement>("#file-input");
const addSourceButton = required<HTMLButtonElement>("#add-source-button");
const sideAddButton = required<HTMLButtonElement>("#side-add-button");
const exportButton = required<HTMLButtonElement>("#export-button");
const clearSessionButton = required<HTMLButtonElement>("#clear-session-button");
const raceButton = required<HTMLButtonElement>("#race-button");
const dropTarget = required<HTMLElement>("#drop-target");
const toastElement = required<HTMLElement>("#toast");
const raceDialog = required<HTMLDialogElement>("#race-dialog");
const raceResult = required<HTMLElement>("#race-result");

interface LoadedSource {
  media: MediaIdentity;
  url: string;
}

const loadedSources = new Map<string, LoadedSource>();
let engine!: ReviewEngine;
let bridge!: VideoFrameBridge;
let unsubscribe: () => void = () => undefined;
let activeMedia: MediaIdentity | null = null;
let scrubbingTimeline = false;
let pendingSeekSeconds: number | null = null;
let renderQueued = false;
let toastTimer = 0;
let restoredDraftMessage: string | null = null;
let lastPersistedSignature = "";
let expectedProgrammaticSeek = false;

const storage = createDraftStorage();

function createDraftStorage(): DraftStorage {
  try {
    const probe = "__speech_review_probe__";
    localStorage.setItem(probe, "1");
    localStorage.removeItem(probe);
    return new LocalStorageDraftStore(localStorage);
  } catch {
    return new MemoryDraftStore();
  }
}

function formatClock(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remaining = safe % 60;
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${remaining
      .toFixed(3)
      .padStart(6, "0")}`;
  }
  return `${minutes.toString().padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function shortId(id: string): string {
  return id.length > 20 ? `${id.slice(0, 17)}…` : id;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function closeReasonLabel(reason: SpeechInterval["closeReason"]): string {
  const labels: Record<SpeechInterval["closeReason"], string> = {
    "key-release": "松键结束",
    "forward-seek": "前向跳转关闭",
    "reverse-seek": "反向跳转关闭",
    "source-change": "换源关闭",
    "window-close": "关窗落盘",
    manual: "手动关闭",
  };
  return labels[reason];
}

function showToast(message: string): void {
  toastElement.textContent = message;
  toastElement.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toastElement.classList.remove("is-visible"), 2800);
}

function configureEngine(
  mode: EngineMode,
  draft: unknown | null,
  mediaToActivate: MediaIdentity | null,
): void {
  unsubscribe();
  bridge?.stop();

  engine = new ReviewEngine({ mode });
  if (draft !== null) {
    engine.restoreDraft(draft);
  }

  activeMedia = mediaToActivate;
  if (activeMedia && loadedSources.has(activeMedia.id)) {
    engine.activateSource(activeMedia);
  }

  bridge = new VideoFrameBridge(video, engine);
  bridge.start();
  unsubscribe = engine.subscribe(handleSnapshot);
  lastPersistedSignature = "";
  expectedProgrammaticSeek = false;
  syncModeChrome(mode);
}

function initializeEngine(): void {
  let draft: unknown | null = null;
  try {
    draft = storage.load();
  } catch {
    draft = null;
  }

  try {
    configureEngine("fixed", draft, null);
    if (draft) {
      const restored = engine.getSnapshot();
      if (restored.sources.length > 0) {
        restoredDraftMessage = `已恢复 ${restored.sources.length} 个来源的草稿；打开原视频即可继续审阅`;
      }
    }
  } catch (error) {
    configureEngine("fixed", null, null);
    restoredDraftMessage = `草稿损坏，已开启空会话：${error instanceof Error ? error.message : "未知错误"}`;
  }
}

function syncModeChrome(mode: EngineMode): void {
  modeSelect.value = mode;
  modeLabel.textContent = mode === "fixed" ? "修复版" : "旧逻辑";
  legacyBanner.hidden = mode !== "legacy";
  document.documentElement.dataset.mode = mode;
}

function handleSnapshot(snapshot: EngineSnapshot): void {
  scheduleRender();
  const signature = [
    snapshot.sources.map((source) => source.id).join(","),
    snapshot.intervals
      .map((interval) => `${interval.id}:${interval.start}:${interval.end}:${interval.closeReason}`)
      .join(","),
  ].join("#");
  if (signature !== lastPersistedSignature) {
    lastPersistedSignature = signature;
    persistDraft();
  }
}

function persistDraft(): void {
  try {
    storage.save(engine.serializeDraft());
  } catch (error) {
    showToast(`草稿写入失败：${error instanceof Error ? error.message : "存储不可用"}`);
  }
}

function scheduleRender(): void {
  if (renderQueued) {
    return;
  }
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render();
  });
}

function render(): void {
  const snapshot = engine.getSnapshot();
  const active = snapshot.activeSource;
  activeMedia = active;
  const isLoaded = active !== null && loadedSources.has(active.id);
  const isLatched = snapshot.latched && snapshot.held;

  videoShell.classList.toggle("has-video", isLoaded);
  activeSourceName.textContent = active?.name ?? "尚未载入视频";
  activeSourceName.title = active?.name ?? "";
  activeSourceId.textContent = active ? active.id : "media —";
  activeSourceId.title = active?.id ?? "";

  if (!active) {
    presentationBadge.textContent = "等待画面";
  } else if (!isLoaded) {
    presentationBadge.textContent = "草稿来源 · 等待重新打开";
  } else if (snapshot.latestPresented) {
    const anchor = snapshot.latestPresented.mediaTime.toFixed(3);
    presentationBadge.textContent = bridge.usesActualPresentation
      ? `rVFC · ${anchor}s`
      : `rAF 近似 · ${anchor}s`;
  } else {
    presentationBadge.textContent = "已载入 · 等待首帧";
  }

  recordingHud.hidden = !snapshot.activeRecording && !isLatched;
  recordingHud.classList.toggle("is-latched", isLatched);
  if (snapshot.activeRecording) {
    recordingHudLabel.textContent = `记录中 · ${snapshot.activeRecording.startAnchor.mediaTime.toFixed(3)}s`;
  } else if (isLatched) {
    recordingHudLabel.textContent = "已闩锁 · 松键后重按";
  }

  const status = recordStatus;
  status.classList.remove("is-active", "is-latched");
  if (snapshot.activeRecording) {
    status.textContent = "记录中";
    status.classList.add("is-active");
  } else if (isLatched) {
    status.textContent = "等待松键";
    status.classList.add("is-latched");
  } else if (snapshot.armed && snapshot.held) {
    status.textContent = "等待首帧";
    status.classList.add("is-active");
  } else {
    status.textContent = snapshot.held ? "已按住" : "待机";
  }

  staleCount.textContent = snapshot.staleCallbacksIgnored.toString();
  intervalCount.textContent = snapshot.intervals.length.toString();
  panelIntervalCount.textContent = snapshot.intervals.length.toString();

  if (!scrubbingTimeline && Number.isFinite(video.duration) && video.duration > 0) {
    timeline.value = Math.round((video.currentTime / video.duration) * 1000).toString();
  }
  currentTimeLabel.textContent = formatClock(video.currentTime);
  durationTimeLabel.textContent = formatClock(
    Number.isFinite(video.duration) ? video.duration : 0,
  );
  playButton.textContent = video.paused ? "▶" : "Ⅱ";
  muteButton.textContent = video.muted ? "静" : "声";

  renderSources(snapshot);
  renderIntervals(snapshot);
  renderJournal(snapshot);

  if (restoredDraftMessage) {
    showToast(restoredDraftMessage);
    restoredDraftMessage = null;
  }
}

function renderSources(snapshot: EngineSnapshot): void {
  const activeId = snapshot.activeSource?.id ?? null;
  const intervalCounts = new Map<string, number>();
  for (const interval of snapshot.intervals) {
    intervalCounts.set(interval.sourceId, (intervalCounts.get(interval.sourceId) ?? 0) + 1);
  }

  const sources = [...snapshot.sources].sort((left, right) => {
    if (left.id === activeId) return -1;
    if (right.id === activeId) return 1;
    const leftLoaded = loadedSources.has(left.id) ? 0 : 1;
    const rightLoaded = loadedSources.has(right.id) ? 0 : 1;
    return leftLoaded - rightLoaded || left.name.localeCompare(right.name);
  });

  if (sources.length === 0) {
    sourceList.innerHTML =
      '<div class="source-empty">打开第一段视频后，媒体指纹与草稿会显示在这里。</div>';
    return;
  }

  sourceList.innerHTML = sources
    .map((source) => {
      const isActive = source.id === activeId;
      const isAvailable = loadedSources.has(source.id);
      const count = intervalCounts.get(source.id) ?? 0;
      return `
        <button class="source-item${isActive ? " is-active" : ""}" type="button"
          data-source-id="${escapeHtml(source.id)}" aria-pressed="${isActive}">
          <span class="source-thumb">${isAvailable ? "▶" : "▱"}</span>
          <span class="source-copy">
            <span class="source-name" title="${escapeHtml(source.name)}">${escapeHtml(source.name)}</span>
            <span class="source-meta" title="${escapeHtml(source.id)}">
              ${formatBytes(source.sizeBytes)} · ${count} 段 · ${escapeHtml(shortId(source.id))}
            </span>
          </span>
          <span class="source-state${isAvailable ? " is-loaded" : ""}">${
            isAvailable ? "已载入" : "仅草稿"
          }</span>
        </button>
      `;
    })
    .join("");
}

function renderIntervals(snapshot: EngineSnapshot): void {
  const sourcesById = new Map(snapshot.sources.map((source) => [source.id, source]));
  const grouped = new Map<string, SpeechInterval[]>();
  for (const interval of snapshot.intervals) {
    const group = grouped.get(interval.sourceId) ?? [];
    group.push(interval);
    grouped.set(interval.sourceId, group);
  }

  const groups: string[] = [];
  for (const source of snapshot.sources) {
    if (grouped.has(source.id)) {
      groups.push(source.id);
    }
  }

  if (snapshot.activeRecording) {
    const active = snapshot.activeRecording;
    const source = sourcesById.get(active.sourceId);
    groups.unshift(`active:${active.intervalId}`);
    const activeRow = `
      <div class="interval-group">
        <div class="interval-group-title">
          <span>${escapeHtml(source?.name ?? active.sourceId)} · 进行中</span>
          <code>待松键/跳转结算</code>
        </div>
        <div class="interval-row">
          <span class="interval-tick"></span>
          <span class="interval-time">${formatClock(active.startAnchor.mediaTime)} → ${formatClock(
            active.lastForwardAnchor.mediaTime,
          )}</span>
          <span class="interval-duration">${
            (
              active.lastForwardAnchor.mediaTime - active.startAnchor.mediaTime
            ).toFixed(3)
          }s</span>
          <span class="interval-reason">锚点来自实际呈现帧 · src ${escapeHtml(
            shortId(active.sourceId),
          )}</span>
        </div>
      </div>
    `;
    groups[0] = activeRow;
  }

  for (const sourceId of groups) {
    if (sourceId.startsWith("active:")) {
      continue;
    }
    const source = sourcesById.get(sourceId);
    const intervals = [...(grouped.get(sourceId) ?? [])].sort(
      (left, right) => left.start - right.start || left.end - right.end,
    );
    groups[groups.indexOf(sourceId)] = `
      <div class="interval-group">
        <div class="interval-group-title" title="${escapeHtml(sourceId)}">
          <span>${escapeHtml(source?.name ?? sourceId)}</span>
          <code>${intervals.length} 段</code>
        </div>
        ${intervals
          .map(
            (interval) => `
              <button class="interval-row" type="button" data-source-id="${escapeHtml(
                interval.sourceId,
              )}" data-start="${interval.start}">
                <span class="interval-tick reason-${interval.closeReason}"></span>
                <span class="interval-time">${formatClock(interval.start)} → ${formatClock(
                  interval.end,
                )}</span>
                <span class="interval-duration">${interval.duration.toFixed(3)}s</span>
                <span class="interval-reason">${escapeHtml(
                  closeReasonLabel(interval.closeReason),
                )} · 源 ${escapeHtml(shortId(interval.sourceId))}</span>
              </button>
            `,
          )
          .join("")}
      </div>
    `;
  }

  intervalList.innerHTML =
    groups.length > 0
      ? groups.join("")
      : '<div class="interval-empty">按住 R 开始记录。区间会绑定到当前媒体身份和实际呈现锚点。</div>';
}

function renderJournal(snapshot: EngineSnapshot): void {
  const entries = snapshot.journal.slice(-14).reverse();
  if (entries.length === 0) {
    journalList.innerHTML = '<li class="journal-item">尚无事件。换源、跳转和迟到的帧会记录在这里。</li>';
    return;
  }
  journalList.innerHTML = entries
    .map((entry) => {
      const isAlert =
        entry.code === "stale-frame-ignored" ||
        entry.code === "reverse-presentation" ||
        entry.code === "interval-discarded";
      const time = new Date(entry.at).toLocaleTimeString("zh-CN", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      return `
        <li class="journal-item${isAlert ? " is-alert" : ""}" title="${escapeHtml(entry.code)}">
          <time>${escapeHtml(time)}</time>
          <span>${escapeHtml(entry.message)}</span>
        </li>
      `;
    })
    .join("");
}

async function addFiles(files: File[]): Promise<void> {
  const videos = files.filter(
    (file) =>
      file.type.startsWith("video/") ||
      /\.(mp4|mkv|mov|webm|m4v|avi)$/i.test(file.name),
  );
  if (videos.length === 0) {
    showToast("请选择视频文件");
    return;
  }

  addSourceButton.disabled = true;
  sideAddButton.disabled = true;
  try {
    for (const file of videos) {
      showToast(`正在计算 ${file.name} 的媒体指纹…`);
      const media = await computeMediaIdentity(file);
      let loaded = loadedSources.get(media.id);
      if (!loaded) {
        loaded = { media, url: URL.createObjectURL(file) };
        loadedSources.set(media.id, loaded);
      }
      await switchToSource(loaded.media);
      showToast(`已载入 ${file.name}`);
    }
  } catch (error) {
    showToast(`载入失败：${error instanceof Error ? error.message : "未知错误"}`);
  } finally {
    addSourceButton.disabled = false;
    sideAddButton.disabled = false;
    fileInput.value = "";
  }
}

async function switchToSource(media: MediaIdentity): Promise<void> {
  const loaded = loadedSources.get(media.id);
  if (!loaded) {
    showToast("该来源只有草稿元数据，请重新打开原视频文件");
    return;
  }

  engine.activateSource(media);
  activeMedia = media;
  expectedProgrammaticSeek = true;
  video.src = loaded.url;
  video.load();
  bridge.invalidate();
  persistDraft();
  scheduleRender();
}

function seekTo(targetSeconds: number): void {
  if (!activeMedia || !Number.isFinite(video.duration) || video.duration <= 0) {
    return;
  }
  const target = Math.min(Math.max(targetSeconds, 0), Math.max(0, video.duration - 0.001));
  engine.requestSeek(target);
  expectedProgrammaticSeek = true;
  video.currentTime = target;
  bridge.invalidate();
  persistDraft();
}

function togglePlayback(): void {
  if (!activeMedia) {
    fileInput.click();
    return;
  }
  if (video.paused) {
    void video.play().catch((error: unknown) => {
      showToast(`无法播放：${error instanceof Error ? error.message : "未知错误"}`);
    });
  } else {
    video.pause();
  }
}

addSourceButton.addEventListener("click", () => fileInput.click());
sideAddButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  void addFiles([...fileInput.files ?? []]);
});

dropTarget.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropTarget.classList.add("is-over");
});
dropTarget.addEventListener("dragleave", () => dropTarget.classList.remove("is-over"));
dropTarget.addEventListener("drop", (event) => {
  event.preventDefault();
  dropTarget.classList.remove("is-over");
  void addFiles([...event.dataTransfer?.files ?? []]);
});
videoShell.addEventListener("dragover", (event) => event.preventDefault());
videoShell.addEventListener("drop", (event) => {
  event.preventDefault();
  void addFiles([...event.dataTransfer?.files ?? []]);
});

sourceList.addEventListener("click", (event) => {
  const button = (event.target as Element).closest<HTMLButtonElement>("[data-source-id]");
  if (!button) {
    return;
  }
  const sourceId = button.dataset.sourceId;
  const media = engine.getSnapshot().sources.find((source) => source.id === sourceId);
  if (media) {
    void switchToSource(media);
  }
});

intervalList.addEventListener("click", (event) => {
  const button = (event.target as Element).closest<HTMLButtonElement>(
    ".interval-row[data-source-id]",
  );
  if (!button) {
    return;
  }
  const sourceId = button.dataset.sourceId;
  const start = Number(button.dataset.start);
  const media = engine.getSnapshot().sources.find((source) => source.id === sourceId);
  if (!media || !Number.isFinite(start)) {
    return;
  }
  if (!loadedSources.has(media.id)) {
    showToast("这段区间来自草稿元数据，请先重新打开原视频");
    return;
  }
  if (activeMedia?.id !== media.id) {
    pendingSeekSeconds = start;
    void switchToSource(media);
  } else {
    seekTo(start);
  }
});

playButton.addEventListener("click", togglePlayback);
video.addEventListener("click", togglePlayback);
backButton.addEventListener("click", () => seekTo(video.currentTime - 5));
forwardButton.addEventListener("click", () => seekTo(video.currentTime + 5));
muteButton.addEventListener("click", () => {
  video.muted = !video.muted;
  scheduleRender();
});

timeline.addEventListener("pointerdown", () => {
  scrubbingTimeline = true;
});
timeline.addEventListener("input", () => {
  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    return;
  }
  const target = (Number(timeline.value) / 1000) * video.duration;
  currentTimeLabel.textContent = formatClock(target);
  seekTo(target);
});
const finishScrub = (): void => {
  scrubbingTimeline = false;
};
timeline.addEventListener("pointerup", finishScrub);
timeline.addEventListener("pointercancel", finishScrub);
timeline.addEventListener("change", finishScrub);

video.addEventListener("loadedmetadata", () => {
  if (pendingSeekSeconds !== null) {
    const target = pendingSeekSeconds;
    pendingSeekSeconds = null;
    seekTo(target);
  }
  bridge.invalidate();
  scheduleRender();
});
video.addEventListener("loadeddata", () => scheduleRender());
video.addEventListener("play", () => scheduleRender());
video.addEventListener("pause", () => scheduleRender());
video.addEventListener("ended", () => scheduleRender());
video.addEventListener("durationchange", () => scheduleRender());
video.addEventListener("seeking", () => {
  if (!expectedProgrammaticSeek) {
    engine.requestSeek(video.currentTime);
    persistDraft();
  }
  expectedProgrammaticSeek = false;
  bridge.invalidate();
});
video.addEventListener("seeked", () => {
  expectedProgrammaticSeek = false;
  bridge.invalidate();
  scheduleRender();
});
video.addEventListener("error", () => {
  showToast("浏览器无法解码该视频；请换用 MP4/WebM 或系统支持的编码");
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "KeyR" || isEditableTarget(event.target) || raceDialog.open) {
    return;
  }
  event.preventDefault();
  engine.handleKeyDown();
});
window.addEventListener("keyup", (event) => {
  if (event.code !== "KeyR") {
    return;
  }
  event.preventDefault();
  engine.handleKeyUp();
});
window.addEventListener("blur", () => {
  if (engine.getSnapshot().held) {
    engine.handleKeyUp();
    persistDraft();
  }
});

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return (
    target.isContentEditable ||
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}

exportButton.addEventListener("click", () => {
  const bundle = engine.exportBundle();
  if (bundle.intervalCount === 0) {
    showToast("还没有可导出的区间");
    return;
  }
  const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `speech-review-${new Date().toISOString().replaceAll(":", "-")}.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast(`已导出 ${bundle.intervalCount} 段区间，含 ${bundle.sourceCount} 个媒体身份`);
});

modeSelect.addEventListener("change", () => {
  const nextMode = modeSelect.value === "legacy" ? "legacy" : "fixed";
  if (nextMode === engine.mode) {
    return;
  }
  const media = activeMedia;
  if (engine.getSnapshot().activeRecording) {
    engine.clearActiveSource();
  }
  const draft = engine.serializeDraft();
  configureEngine(nextMode, draft, media && loadedSources.has(media.id) ? media : null);
  bridge.invalidate();
  persistDraft();
  showToast(nextMode === "fixed" ? "已切到修复版会话" : "已切到旧逻辑复现会话");
});

clearSessionButton.addEventListener("click", () => {
  if (!window.confirm("清空全部区间、媒体身份和本地草稿？此操作不可撤销。")) {
    return;
  }
  for (const loaded of loadedSources.values()) {
    URL.revokeObjectURL(loaded.url);
  }
  loadedSources.clear();
  video.removeAttribute("src");
  video.load();
  activeMedia = null;
  configureEngine(engine.mode, null, null);
  storage.clear();
  showToast("会话已清空");
});

raceButton.addEventListener("click", () => {
  renderRaceComparison();
  raceDialog.showModal();
});
required<HTMLButtonElement>("#race-close").addEventListener("click", () => raceDialog.close());
required<HTMLButtonElement>("#race-rerun").addEventListener("click", renderRaceComparison);

function renderRaceComparison(): void {
  const comparison = runDelayedSwitchRace();
  const fixed = comparison.fixed.betaIntervals[0];
  const legacy = comparison.legacy.betaIntervals[0];
  const describe = (interval: SpeechInterval | undefined): string =>
    interval
      ? `${formatClock(interval.start)} → ${formatClock(interval.end)}`
      : "无区间";
  raceResult.innerHTML = `
    <div class="race-side is-fixed">
      <span class="race-side-label">FIXED</span>
      <strong>${describe(fixed)}</strong>
      <p>拦截 ${comparison.fixed.staleCallbacksIgnored} 个迟到帧；B 源只保留自身实际呈现位置。</p>
    </div>
    <div class="race-side is-legacy">
      <span class="race-side-label">LEGACY</span>
      <strong>${describe(legacy)}</strong>
      <p>拦截 ${comparison.legacy.staleCallbacksIgnored} 个迟到帧；A 源的 12s 帧被写成 B 源区间终点。</p>
    </div>
  `;
}

let closingFlushed = false;
function flushBeforeClose(): void {
  if (closingFlushed) {
    return;
  }
  closingFlushed = true;
  const draft = engine.handleWindowClose();
  try {
    storage.save(draft);
  } catch {}
  bridge.stop();
}
window.addEventListener("pagehide", flushBeforeClose);
window.addEventListener("beforeunload", flushBeforeClose);

function runInterfaceClock(): void {
  if (!scrubbingTimeline) {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      timeline.value = Math.round((video.currentTime / video.duration) * 1000).toString();
    }
    currentTimeLabel.textContent = formatClock(video.currentTime);
    durationTimeLabel.textContent = formatClock(
      Number.isFinite(video.duration) ? video.duration : 0,
    );
  }
  requestAnimationFrame(runInterfaceClock);
}

initializeEngine();
scheduleRender();
runInterfaceClock();
