import { timeStretchChannels } from "./stretch.js";

self.onmessage = (event) => {
  const { id, channels, ratio, sampleRate } = event.data;
  try {
    const startedAt = performance.now();
    const output = timeStretchChannels(channels, ratio, sampleRate);
    self.postMessage(
      {
        id,
        output,
        elapsed: performance.now() - startedAt,
      },
      output.map((channel) => channel.buffer),
    );
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
