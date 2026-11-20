import { buildTimeline, countInPulses, retimeFuture, roundAtTime, roundPulses } from "./timeline.js";

const LOOKAHEAD = 0.55;

function createClick(context, destination, time, options = {}) {
  const { accent = false, countIn = false, volume = 0.12 } = options;
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  const duration = countIn ? 0.055 : 0.032;
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(countIn ? (accent ? 1760 : 1320) : accent ? 1480 : 1040, time);
  gain.gain.setValueAtTime(0.0001, time);
  gain.gain.exponentialRampToValueAtTime(volume * (accent ? 1 : 0.62), time + 0.002);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + duration);
  oscillator.connect(gain).connect(destination);
  oscillator.start(time);
  oscillator.stop(time + duration + 0.01);
  return oscillator;
}

function ydBToGain(value) {
  return 10 ** (value / 20);
}

export class LoopAudioEngine {
  constructor() {
    this.context = null;
    this.master = null;
    this.clickBus = null;
    this.worker = null;
    this.workerJobs = new Map();
    this.jobId = 0;
    this.track = null;
    this.trackId = "";
    this.loopIn = 0;
    this.loopOut = 1;
    this.segmentCache = new Map();
    this.pendingSegments = new Map();
    this.timeline = null;
    this.playing = false;
    this.scheduled = new Map();
    this.countInNodes = [];
    this.timer = null;
    this.nextRound = 0;
    this.clickEnabled = true;
    this.lookahead = LOOKAHEAD;
    this.auditionNode = null;
  }

  async ensureContext(resume = false) {
    if (!this.context) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.context = new AudioContextClass({ latencyHint: "interactive" });
      this.master = this.context.createGain();
      this.clickBus = this.context.createGain();
      this.master.gain.value = ydBToGain(-6);
      this.master.connect(this.context.destination);
      this.clickBus.gain.value = 0.72;
      this.clickBus.connect(this.master);
      this.worker = new Worker(new URL("./stretch-worker.js", import.meta.url), { type: "module" });
      this.worker.onmessage = (event) => this.handleWorkerMessage(event);
    }
    if (resume && this.context.state !== "running") await this.context.resume();
    return this.context;
  }

  handleWorkerMessage(event) {
    const { id, output, error, elapsed } = event.data;
    const job = this.workerJobs.get(id);
    if (!job) return;
    this.workerJobs.delete(id);
    if (error) {
      job.reject(new Error(error));
      return;
    }
    const buffer = this.context.createBuffer(output.length, output[0].length, job.sampleRate);
    output.forEach((channel, index) => buffer.copyToChannel(channel, index));
    job.resolve({ buffer, elapsed });
  }

  setMasterVolume(value) {
    if (!this.master) return;
    this.master.gain.setTargetAtTime(ydBToGain(value), this.context.currentTime, 0.02);
  }

  setClickEnabled(enabled) {
    this.clickEnabled = enabled;
    if (this.clickBus) {
      this.clickBus.gain.setTargetAtTime(enabled ? 0.72 : 0, this.context.currentTime, 0.015);
    }
  }

  cacheKey(bpm, trackId = this.trackId) {
    return `${trackId}:${this.loopIn.toFixed(6)}:${this.loopOut.toFixed(6)}:${bpm}`;
  }

  async loadTrack(track, url) {
    await this.ensureContext();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}`);
    const data = await response.arrayBuffer();
    const buffer = await this.context.decodeAudioData(data);
    this.stop();
    this.track = { ...track, buffer, url };
    this.trackId = track.id;
    this.segmentCache.clear();
    this.pendingSegments.clear();
    return this.track;
  }

  setLoop(sourceIn, sourceOut) {
    this.loopIn = sourceIn;
    this.loopOut = sourceOut;
    this.segmentCache.clear();
    this.pendingSegments.clear();
  }

  extractSegment() {
    if (!this.track) throw new Error("Load a track before preparing audio.");
    const buffer = this.track.buffer;
    const sampleRate = buffer.sampleRate;
    const begin = Math.max(0, Math.round(this.loopIn * sampleRate));
    const end = Math.min(buffer.length, Math.round(this.loopOut * sampleRate));
    return Array.from({ length: buffer.numberOfChannels }, (_, channel) => {
      return buffer.getChannelData(channel).slice(begin, end);
    });
  }

  renderTempo(bpm) {
    const key = this.cacheKey(bpm);
    if (this.segmentCache.has(key)) return Promise.resolve(this.segmentCache.get(key));
    if (this.pendingSegments.has(key)) return this.pendingSegments.get(key);

    const promise = new Promise(async (resolve, reject) => {
      try {
        await this.ensureContext();
        const sourceBpm = this.track.bpm;
        const sourceRate = this.track.buffer.sampleRate;
        const ratio = bpm / sourceBpm;
        if (Math.abs(ratio - 1) < 1e-5) {
          const segment = this.extractSegment();
          const buffer = this.context.createBuffer(
            segment.length,
            segment[0].length,
            sourceRate,
          );
          segment.forEach((channel, index) => buffer.copyToChannel(channel, index));
          const result = { buffer, elapsed: 0 };
          this.segmentCache.set(key, result);
          resolve(result);
          return;
        }

        const channels = this.extractSegment();
        const id = ++this.jobId;
        this.workerJobs.set(id, { resolve, reject, sampleRate: sourceRate });
        this.worker.postMessage(
          { id, channels, ratio, sampleRate: sourceRate },
          channels.map((channel) => channel.buffer),
        );
      } catch (error) {
        reject(error);
      }
    }).then((result) => {
      this.segmentCache.set(key, result);
      this.pendingSegments.delete(key);
      return result;
    });

    this.pendingSegments.set(key, promise);
    promise.catch(() => this.pendingSegments.delete(key));
    return promise;
  }

  async prepare(ladder) {
    const unique = [...new Set(ladder)];
    const results = [];
    for (const bpm of unique) {
      results.push(await this.renderTempo(bpm));
    }
    return results;
  }

  async audition(bpm) {
    await this.ensureContext(true);
    this.stop();
    const result = await this.renderTempo(bpm);
    const source = this.context.createBufferSource();
    const gain = this.context.createGain();
    source.buffer = result.buffer;
    source.connect(gain).connect(this.master);
    gain.gain.value = 0.92;
    const startAt = this.context.currentTime + 0.04;
    source.start(startAt);
    source.stop(startAt + result.buffer.duration);
    this.auditionNode = source;
    return { duration: result.buffer.duration, startAt };
  }

  getSegment(bpm) {
    return this.segmentCache.get(this.cacheKey(bpm))?.buffer ?? null;
  }

  scheduleClickSet(pulses, roundIndex = null) {
    if (!this.clickEnabled) return;
    const nodes = [];
    for (const pulse of pulses) {
      if (pulse.time < this.context.currentTime - 0.01) continue;
      const node = createClick(this.context, this.clickBus, pulse.time, {
        accent: pulse.downbeat,
        countIn: pulse.countIn,
        volume: pulse.countIn ? 0.15 : 0.09,
      });
      nodes.push(node);
    }
    if (roundIndex !== null) this.scheduled.set(`click-${roundIndex}`, nodes);
    else this.countInNodes.push(...nodes);
  }

  scheduleRound(round) {
    const buffer = this.getSegment(round.bpm);
    if (!buffer) return false;
    if (round.end <= this.context.currentTime - 0.01) return true;
    const source = this.context.createBufferSource();
    const gain = this.context.createGain();
    source.buffer = buffer;
    source.connect(gain).connect(this.master);
    const startAt = Math.max(this.context.currentTime, round.start);
    const offset = Math.max(0, this.context.currentTime - round.start);
    gain.gain.setValueAtTime(1, startAt);
    source.start(startAt, offset);
    source.stop(Math.max(startAt + 0.02, round.end));
    this.scheduled.set(`audio-${round.index}`, { source, gain });
    if (this.clickEnabled) this.scheduleClickSet(roundPulses(this.timeline, round.index), round.index);
    return true;
  }

  async play({ ladder, countInBars = 1, beatsPerBar = 4, lookahead = LOOKAHEAD }) {
    await this.ensureContext(true);
    this.stop();
    await this.prepare(ladder);
    this.lookahead = lookahead;
    const anchorTime = this.context.currentTime + 0.09;
    this.timeline = buildTimeline({
      anchorTime,
      sourceIn: this.loopIn,
      sourceOut: this.loopOut,
      sourceBpm: this.track.bpm,
      ladder,
      countInBars,
      beatsPerBar,
      roundCount: 256,
    });
    this.playing = true;
    this.nextRound = 0;
    this.scheduleClickSet(countInPulses(this.timeline));
    this.tick();
    this.timer = setInterval(() => this.tick(), 60);
    return this.timeline;
  }

  tick() {
    if (!this.playing) return;
    const now = this.context.currentTime;
    while (
      this.nextRound < this.timeline.rounds.length &&
      this.timeline.rounds[this.nextRound].start <= now + this.lookahead
    ) {
      const round = this.timeline.rounds[this.nextRound];
      if (!this.scheduleRound(round)) break;
      this.nextRound += 1;
    }
  }

  retime(nextLadder) {
    if (!this.playing || !this.timeline) return null;
    const now = this.context.currentTime;
    if (now < this.timeline.loopStart) {
      this.cancelFrom(0);
      this.timeline = retimeFuture(this.timeline, 0, nextLadder);
      this.nextRound = 0;
      this.tick();
      return this.timeline;
    }
    const current = roundAtTime(this.timeline, now);
    if (!current) return this.timeline;
    const fromRound = current.index + 1;
    this.cancelFrom(fromRound);
    this.timeline = retimeFuture(this.timeline, fromRound, nextLadder);
    this.nextRound = fromRound;
    this.tick();
    return this.timeline;
  }

  cancelFrom(roundIndex) {
    for (const key of [...this.scheduled.keys()]) {
      const separator = key.lastIndexOf("-");
      const index = Number(key.slice(separator + 1));
      if (index < roundIndex) continue;
      const value = this.scheduled.get(key);
      const nodes = Array.isArray(value) ? value : [value?.source].filter(Boolean);
      for (const node of nodes) {
        try {
          node.stop();
        } catch {
          // A node may already have ended.
        }
      }
      this.scheduled.delete(key);
    }
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.cancelFrom(0);
    for (const node of this.countInNodes) {
      try {
        node.stop();
      } catch {
        // Already ended.
      }
    }
    this.countInNodes = [];
    if (this.auditionNode) {
      try {
        this.auditionNode.stop();
      } catch {
        // Preview already finished.
      }
      this.auditionNode = null;
    }
    this.playing = false;
    this.nextRound = 0;
  }

  now() {
    return this.context?.currentTime ?? 0;
  }
}
