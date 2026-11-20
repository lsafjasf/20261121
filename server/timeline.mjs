const MAX_POSITION_MS = 24 * 3600 * 1000;

export class Timeline {
  constructor(cuesA, cuesB) {
    this.cuesA = cuesA;
    this.cuesB = cuesB;
    this.state = {
      playing: false,
      anchorPos: 0,
      anchorServerMs: null,
      delayMs: 0,
      clearUntil: 0,
      clearFrom: null,
      updatedAt: null,
    };
    this.seq = 0;
  }

  now() {
    return Date.now();
  }

  positionAt(serverMs = this.now()) {
    const s = this.state;
    if (!s.playing || s.anchorServerMs === null) return s.anchorPos;
    return s.anchorPos + Math.max(0, serverMs - s.anchorServerMs);
  }

  reanchor(serverMs = this.now()) {
    this.state.anchorPos = this.positionAt(serverMs);
    this.state.anchorServerMs = serverMs;
  }

  snapshot(serverMs = this.now()) {
    const s = this.state;
    const clearUntil = s.clearUntil === Infinity ? null : s.clearUntil;
    return {
      seq: this.seq,
      playing: s.playing,
      anchorPos: s.anchorPos,
      anchorServerMs: s.anchorServerMs,
      delayMs: s.delayMs,
      clearUntil,
      clearFrom: s.clearFrom,
      positionMs: this.positionAt(serverMs),
      serverMs,
      cuesA: this.cuesA,
      cuesB: this.cuesB,
    };
  }

  apply(action, payload = {}) {
    const now = this.now();
    const s = this.state;
    const move = (pos) => {
      s.anchorPos = Math.min(MAX_POSITION_MS, Math.max(0, Math.round(pos)));
      s.anchorServerMs = s.playing ? now : null;
    };

    switch (action) {
      case 'play':
        if (!s.playing) {
          s.playing = true;
          s.anchorServerMs = now;
        }
        break;
      case 'pause':
        this.reanchor(now);
        s.playing = false;
        s.anchorServerMs = null;
        break;
      case 'seek': {
        const target = Number(payload.positionMs ?? this.state.anchorPos);
        if (!Number.isFinite(target)) break;
        move(target);
        break;
      }
      case 'nudge': {
        const nudgeDelta = Number(payload.deltaMs ?? 0);
        if (!Number.isFinite(nudgeDelta)) break;
        move(this.positionAt(now) + nudgeDelta);
        break;
      }
      case 'nextCue':
        move(this.cueAfter(this.positionAt(now)));
        break;
      case 'prevCue':
        move(this.cueBefore(this.positionAt(now)));
        break;
      case 'setDelay': {
        const delay = Number(payload.delayMs ?? 0);
        if (!Number.isFinite(delay)) break;
        s.delayMs = Math.max(0, Math.round(delay));
        break;
      }
      case 'nudgeDelay': {
        const delayDelta = Number(payload.delayMs ?? 0);
        if (!Number.isFinite(delayDelta)) break;
        s.delayMs = Math.max(0, s.delayMs + Math.round(delayDelta));
        break;
      }
      case 'clear': {
        const rawHoldMs = payload.holdMs;
        const holdMs = rawHoldMs === undefined || rawHoldMs === null
          ? null : Math.max(0, Number(rawHoldMs));
        if (holdMs !== null && !Number.isFinite(holdMs)) break;
        s.clearFrom = now;
        s.clearUntil = holdMs === null ? Infinity : now + holdMs;
        break;
      }
      case 'unclear':
        s.clearFrom = null;
        s.clearUntil = 0;
        break;
      default:
        return false;
    }
    this.seq += 1;
    s.updatedAt = now;
    return true;
  }

  cueAfter(position) {
    const later = this.cuesA.find((c) => c.start > position + 1);
    return later ? later.start : position;
  }

  cueBefore(position) {
    const earlier = [...this.cuesA].reverse().find((c) => c.start < position - 1);
    return earlier ? earlier.start : 0;
  }
}
