export const TRACKS = [
  {
    id: "blue-hour",
    title: "Blue Hour",
    subtitle: "磁带爵士 · 电钢与行走贝斯",
    style: "JAZZ",
    bpm: 88,
    key: "F major",
    meter: "4 / 4",
    bars: 8,
    file: "samples/blue-hour.wav",
    accent: "#d7ff69",
    defaultBeats: [4, 12],
    description: "Warm, swung phrasing with a wide pocket for articulation work.",
  },
  {
    id: "neon-current",
    title: "Neon Current",
    subtitle: "极简电子 · 琶音与紧实底鼓",
    style: "ELECTRONIC",
    bpm: 124,
    key: "A minor",
    meter: "4 / 4",
    bars: 8,
    file: "samples/neon-current.wav",
    accent: "#ffb85f",
    defaultBeats: [0, 8],
    description: "Tight electronic pulses for precision, endurance and clean attacks.",
  },
];

export const LADDER_PRESETS = {
  warmup: [64, 68, 72, 76, 80, 84, 88, 92, 96, 100],
  build: [72, 76, 80, 84, 88, 92, 96, 100, 104, 108],
  polish: [116, 112, 108, 104, 100, 96, 92, 88, 84, 80],
};
