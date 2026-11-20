// AudioWorklet: ACN/SN3D 四路输入 -> 头部旋转 -> 立体声输出。
// 与 foa.js 相同的旋转约定（worklet 为经典脚本，内联所需数学）。
const DEG = Math.PI / 180;
const CAPTURE_LEN = 2048;

class FOAListenerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'yaw', defaultValue: 0, minValue: -1440, maxValue: 1440, automationRate: 'k-rate' },
      { name: 'pitch', defaultValue: 0, minValue: -90, maxValue: 90, automationRate: 'k-rate' },
      { name: 'roll', defaultValue: 0, minValue: -180, maxValue: 180, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this.sums = new Float64Array(6);
    this.blocks = 0;
    this.refL = null;
    this.refR = null;
    this.bufL = null;
    this.bufR = null;
    this.fill = 0;
    this.mode = null; // 'capture' | 'compare'
    this.port.onmessage = (e) => {
      if (e.data === 'capture' || e.data === 'compare') {
        this.bufL = new Float32Array(CAPTURE_LEN);
        this.bufR = new Float32Array(CAPTURE_LEN);
        this.fill = 0;
        this.mode = e.data;
      }
    };
  }

  process(inputs, outputs, params) {
    const input = inputs[0];
    const outL = outputs[0][0];
    const outR = outputs[0][1];
    if (!input || input.length < 4 || input[0].length === 0) {
      outL.fill(0);
      outR.fill(0);
      return true;
    }
    const yaw = params.yaw[0] * DEG;
    const pitch = params.pitch[0] * DEG;
    const roll = params.roll[0] * DEG;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    // M = Rx(-roll) * Ry(pitch) * Rz(-yaw)，展开为 9 个系数
    const m00 = cp * cy, m01 = cp * sy, m02 = sp;
    const m10 = -cr * sy - sr * sp * cy, m11 = cr * cy - sr * sp * sy, m12 = sr * cp;
    const m20 = sr * sy - cr * sp * cy, m21 = -sr * cy - cr * sp * sy, m22 = cr * cp;
    const c30 = Math.cos(30 * DEG), s30 = Math.sin(30 * DEG);

    const W = input[0], Y = input[1], Z = input[2], X = input[3];
    const n = outL.length;
    let sW = 0, sY = 0, sZ = 0, sX = 0, sL = 0, sR = 0;
    for (let i = 0; i < n; i++) {
      const x = m00 * X[i] + m01 * Y[i] + m02 * Z[i];
      const y = m10 * X[i] + m11 * Y[i] + m12 * Z[i];
      const z = m20 * X[i] + m21 * Y[i] + m22 * Z[i];
      const w = W[i];
      const l = 0.5 * (w + c30 * x + s30 * y);
      const r = 0.5 * (w + c30 * x - s30 * y);
      outL[i] = l;
      outR[i] = r;
      sW += w * w; sY += y * y; sZ += z * z; sX += x * x; sL += l * l; sR += r * r;
    }

    if (this.mode) this.record(outL, outR);

    this.sums[0] += sW; this.sums[1] += sY; this.sums[2] += sZ;
    this.sums[3] += sX; this.sums[4] += sL; this.sums[5] += sR;
    if (++this.blocks >= 10) {
      const total = this.blocks * n;
      const rms = [];
      for (let k = 0; k < 6; k++) rms.push(Math.sqrt(this.sums[k] / total));
      this.port.postMessage({ type: 'rms', rms });
      this.sums.fill(0);
      this.blocks = 0;
    }
    return true;
  }

  record(outL, outR) {
    const n = Math.min(outL.length, CAPTURE_LEN - this.fill);
    this.bufL.set(outL.subarray(0, n), this.fill);
    this.bufR.set(outR.subarray(0, n), this.fill);
    this.fill += n;
    if (this.fill < CAPTURE_LEN) return;
    if (this.mode === 'capture') {
      this.refL = this.bufL;
      this.refR = this.bufR;
      this.port.postMessage({ type: 'captured' });
    } else if (this.refL) {
      let maxDiff = 0;
      for (let i = 0; i < CAPTURE_LEN; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(this.bufL[i] - this.refL[i]), Math.abs(this.bufR[i] - this.refR[i]));
      }
      this.port.postMessage({ type: 'diff', value: maxDiff });
    }
    this.mode = null;
    this.bufL = this.bufR = null;
  }
}

registerProcessor('foa-listener', FOAListenerProcessor);
