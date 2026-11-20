import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SR = 44100;
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function mulberry32(seed) {
  return () => {
    let value = (seed += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makeTrack(seconds) {
  return new Float32Array(Math.ceil(seconds * SR));
}

function envelope(time, attack, decay, release = 0.04) {
  if (time < 0) return 0;
  if (time < attack) return time / attack;
  if (time < attack + decay) {
    const progress = (time - attack) / decay;
    return 1 - 0.72 * progress;
  }
  const tail = Math.max(0, 1 - (time - attack - decay) / release);
  return 0.28 * tail * tail;
}

function addTone(track, start, duration, frequency, options = {}) {
  const {
    amp = 0.1,
    attack = 0.006,
    decay = duration * 0.65,
    release = 0.08,
    partials = [1],
    detune = 0,
    vibrato = 0,
  } = options;
  const begin = Math.max(0, Math.floor(start * SR));
  const end = Math.min(track.length, Math.floor((start + duration) * SR));
  for (let sample = begin; sample < end; sample += 1) {
    const time = sample / SR - start;
    const wobble = vibrato ? Math.sin(2 * Math.PI * 5.2 * time) * vibrato : 0;
    let value = 0;
    for (let index = 0; index < partials.length; index += 1) {
      const ratio = index + 1;
      value +=
        partials[index] *
        Math.sin(
          2 * Math.PI * frequency * ratio * time +
            Math.sin(2 * Math.PI * (0.17 + index * 0.03) * time) * 0.08 +
            wobble,
        );
    }
    track[sample] += value * amp * envelope(time, attack, decay, release);
  }
}

function addPluck(track, start, frequency, amp, duration = 0.55, seed = 1) {
  const random = mulberry32(seed);
  const count = Math.floor(duration * SR);
  const begin = Math.floor(start * SR);
  const ring = new Float32Array(96);
  ring.fill(random() * 2 - 1);
  let cursor = 0;
  for (let index = 0; index < count; index += 1) {
    const sampleIndex = begin + index;
    if (sampleIndex >= track.length) break;
    const current = ring[cursor];
    const next = ring[(cursor + 1) % ring.length];
    const filtered = (current + next) * 0.497;
    ring[cursor] = filtered;
    cursor = (cursor + 1) % ring.length;
    track[sampleIndex] += current * amp * Math.exp((-5.5 * index) / count);
  }
  addTone(track, start, Math.min(0.4, duration), frequency, {
    amp: amp * 0.25,
    partials: [1, 0.35, 0.12],
    decay: 0.24,
  });
}

function addNoise(track, start, duration, amp, filterAmount, seed) {
  const random = mulberry32(seed);
  const begin = Math.max(0, Math.floor(start * SR));
  const count = Math.floor(duration * SR);
  let previous = 0;
  for (let index = 0; index < count && begin + index < track.length; index += 1) {
    const white = random() * 2 - 1;
    const filtered = white - previous * filterAmount;
    previous = white;
    const decay = Math.exp((-7 * index) / count);
    track[begin + index] += filtered * amp * decay;
  }
}

function addKick(track, start, amp = 0.5) {
  const begin = Math.floor(start * SR);
  const count = Math.floor(0.38 * SR);
  let phase = 0;
  for (let index = 0; index < count && begin + index < track.length; index += 1) {
    const time = index / SR;
    const frequency = 46 + 105 * Math.exp(-time * 28);
    phase += (2 * Math.PI * frequency) / SR;
    const body = Math.sin(phase) * Math.exp(-time * 9);
    const click = index < 90 ? (Math.random() * 2 - 1) * 0.13 * (1 - index / 90) : 0;
    track[begin + index] += (body + click) * amp;
  }
}

function normalize(track, peak = 0.88) {
  let maximum = 0;
  for (const sample of track) maximum = Math.max(maximum, Math.abs(sample));
  if (!maximum) return track;
  const gain = peak / maximum;
  for (let index = 0; index < track.length; index += 1) track[index] *= gain;
  return track;
}

function writeWav(path, track) {
  const buffer = Buffer.alloc(44 + track.length * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + track.length * 2, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(SR, 24);
  buffer.writeUInt32LE(SR * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(track.length * 2, 40);
  for (let index = 0; index < track.length; index += 1) {
    const value = Math.max(-1, Math.min(1, track[index]));
    buffer.writeInt16LE(Math.round(value * 32767), 44 + index * 2);
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, buffer);
}

const note = (midi) => 440 * 2 ** ((midi - 69) / 12);
export { SR, root, makeTrack, addTone, addPluck, addNoise, addKick, normalize, writeWav, note };

function buildBlueHour() {
  const bpm = 88;
  const bars = 8;
  const beat = 60 / bpm;
  const track = makeTrack(bars * 4 * beat);
  const progression = [
    { root: 41, notes: [53, 57, 60, 64] },
    { root: 38, notes: [50, 53, 57, 60] },
    { root: 43, notes: [55, 58, 62, 65] },
    { root: 36, notes: [48, 52, 55, 58] },
  ];

  for (let bar = 0; bar < bars; bar += 1) {
    const chord = progression[bar % progression.length];
    const barStart = bar * 4 * beat;
    const walks = [0, 4, 7, 10];
    const walkingRoots = [
      chord.root,
      chord.root + (bar % 2 ? 5 : 4),
      chord.root + 7,
      chord.root + (bar % 3 === 0 ? 11 : 10),
    ];

    for (let quarter = 0; quarter < 4; quarter += 1) {
      addTone(track, barStart + quarter * beat, beat * 1.7, note(walkingRoots[quarter]), {
        amp: 0.16,
        attack: 0.012,
        decay: beat * 0.8,
        partials: [1, 0.32, 0.1],
      });
      addNoise(track, barStart + quarter * beat, 0.13, 0.04, 0.62, 100 + bar * 8 + quarter);
      const swung = barStart + (quarter + 2 / 3) * beat;
      addNoise(track, swung, 0.085, 0.025, 0.75, 800 + bar * 8 + quarter);
    }

    [0, 1.68, 2.72].forEach((offset, hit) => {
      chord.notes.forEach((midi, voice) => {
        addTone(track, barStart + offset * beat, beat * 1.15, note(midi), {
          amp: 0.052 - voice * 0.004,
          attack: 0.008,
          decay: beat * 0.8,
          partials: [1, 0.5, 0.2, 0.08],
          detune: hit % 2 ? 1.5 : -1.5,
        });
      });
    });

    addNoise(track, barStart + beat, 0.2, 0.055, 0.85, 300 + bar);
    addNoise(track, barStart + 3 * beat, 0.2, 0.05, 0.85, 500 + bar);

    const guide = [chord.notes[2], chord.notes[1], chord.notes[3], chord.notes[0]];
    addTone(track, barStart + 2.5 * beat, beat * 1.2, note(guide[bar % 4] + 12), {
      amp: 0.055,
      attack: 0.03,
      decay: beat * 1.05,
      partials: [1, 0.22, 0.1],
      vibrato: 0.006,
    });
  }
  return normalize(track, 0.86);
}

function buildNeonCurrent() {
  const bpm = 124;
  const bars = 8;
  const beat = 60 / bpm;
  const sixteenth = beat / 4;
  const track = makeTrack(bars * 4 * beat);
  const roots = [45, 41, 48, 43];
  const arpSteps = [0, 7, 12, 15, 19, 15, 12, 7];

  for (let bar = 0; bar < bars; bar += 1) {
    const root = roots[bar % roots.length];
    const barStart = bar * 4 * beat;
    for (let quarter = 0; quarter < 4; quarter += 1) {
      addKick(track, barStart + quarter * beat, quarter === 0 ? 0.58 : 0.48);
      if (quarter === 1 || quarter === 3) {
        addNoise(track, barStart + quarter * beat, 0.17, 0.22, 0.78, 2000 + bar * 4 + quarter);
      }
    }

    for (let step = 0; step < 16; step += 1) {
      const time = barStart + step * sixteenth;
      if (step % 4 === 2) {
        addNoise(track, time, 0.13, 0.12, 0.68, 4000 + bar * 16 + step);
      } else if (step % 2 === 0) {
        addNoise(track, time, 0.035, 0.065, 0.9, 5000 + bar * 16 + step);
      }

      if ([0, 2, 5, 7, 10, 12, 14].includes(step)) {
        addTone(track, time, sixteenth * 1.8, note(root - 12), {
          amp: step % 4 === 0 ? 0.2 : 0.14,
          attack: 0.004,
          decay: sixteenth * 1.2,
          partials: [1, 0.3, 0.08],
        });
      }

      const arpNote = root + 12 + arpSteps[step % arpSteps.length];
      addPluck(track, time, note(arpNote), step % 4 === 0 ? 0.15 : 0.105, 0.5, 50 + bar * 16 + step);
    }

    [0, 2.5].forEach((offset, index) => {
      addTone(track, barStart + offset * beat, beat * 1.9, note(root + [3, 10][index]), {
        amp: 0.045,
        attack: 0.12,
        decay: beat * 1.3,
        partials: [1, 0.35, 0.15, 0.06],
      });
    });
  }

  const echo = new Float32Array(track.length);
  const delay = Math.floor(beat * 0.75 * SR);
  for (let index = delay; index < track.length; index += 1) {
    echo[index] = track[index - delay] * 0.19;
  }
  for (let index = 0; index < track.length; index += 1) track[index] += echo[index];
  return normalize(track, 0.88);
}

const output = process.argv[2] ? resolve(process.argv[2]) : resolve(root, "samples");
writeWav(resolve(output, "blue-hour.wav"), buildBlueHour());
writeWav(resolve(output, "neon-current.wav"), buildNeonCurrent());
console.log(`Generated sample tracks in ${output}`);
