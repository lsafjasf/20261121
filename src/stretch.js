const TAU = Math.PI * 2;

function hann(length) {
  const window = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    window[index] = 0.5 - 0.5 * Math.cos((TAU * index) / (length - 1));
  }
  return window;
}

function bestOffset(input, desired, size, tolerance, step, referenceStart) {
  const min = Math.max(0, Math.round(desired - tolerance));
  const max = Math.min(input.length - size, Math.round(desired + tolerance));
  const reference = Math.max(0, Math.round(referenceStart));
  let best = Math.max(0, Math.min(input.length - size, Math.round(desired)));
  let bestScore = -Infinity;

  for (let candidate = min; candidate <= max; candidate += step) {
    let score = 0;
    for (let index = 0; index < size; index += 4) {
      const referenceIndex = reference + index;
      if (referenceIndex >= input.length) break;
      score += input[candidate + index] * input[referenceIndex];
    }
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return best;
}

export function timeStretchChannel(input, ratio, sampleRate = 44100, options = {}) {
  if (!(ratio > 0)) throw new Error("Stretch ratio must be positive.");
  const frameSize = options.frameSize ?? 2048;
  const synthesisHop = options.synthesisHop ?? frameSize / 2;
  const tolerance = options.tolerance ?? Math.round(sampleRate * 0.01);
  const searchStep = options.searchStep ?? 4;
  const outputLength = Math.max(1, Math.round(input.length / ratio));

  if (Math.abs(ratio - 1) < 1e-5 || input.length <= frameSize * 2) {
    const copy = new Float32Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      const source = Math.min(input.length - 1, index * ratio);
      const left = Math.floor(source);
      const mix = source - left;
      copy[index] = input[left] * (1 - mix) + (input[Math.min(input.length - 1, left + 1)] ?? 0) * mix;
    }
    return copy;
  }

  const output = new Float32Array(outputLength + frameSize);
  const weight = new Float32Array(outputLength + frameSize);
  const window = hann(frameSize);
  const analysisHop = synthesisHop * ratio;
  let outputPosition = 0;
  let frameIndex = 0;
  let previousAnalysis = 0;

  while (outputPosition < outputLength) {
    const desired = Math.min(input.length - frameSize, frameIndex * analysisHop);
    let analysis = Math.max(0, Math.round(desired));
    if (frameIndex > 0) {
      analysis = bestOffset(
        input,
        desired,
        synthesisHop,
        tolerance,
        searchStep,
        previousAnalysis + synthesisHop,
      );
    }
    for (let index = 0; index < frameSize; index += 1) {
      const destination = outputPosition + index;
      if (destination >= output.length) break;
      const value = (input[analysis + index] ?? 0) * window[index];
      output[destination] += value;
      weight[destination] += window[index];
    }
    previousAnalysis = analysis;
    outputPosition += synthesisHop;
    frameIndex += 1;
  }

  const result = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    result[index] = weight[index] > 0.02 ? output[index] / weight[index] : output[index];
  }

  const edge = Math.min(Math.round(sampleRate * 0.003), Math.floor(outputLength / 8));
  for (let index = 0; index < edge; index += 1) {
    const gain = index / edge;
    result[index] *= gain;
    result[outputLength - 1 - index] *= gain;
  }
  return result;
}

export function timeStretchChannels(channels, ratio, sampleRate = 44100, options = {}) {
  return channels.map((channel) => timeStretchChannel(channel, ratio, sampleRate, options));
}
