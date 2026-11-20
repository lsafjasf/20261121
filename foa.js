// 一阶 Ambisonics（ACN/SN3D, AmbiX）共享数学模块。
// 通道顺序 ACN: [W, Y, Z, X]。坐标系: X=前, Y=左, Z=上（右手系）。
// 头部方位约定: yaw>0 向左转, pitch>0 抬头, roll>0 右耳下沉。

export const DEG = Math.PI / 180;

// 单声道信号从 (azimuthDeg, elevationDeg) 方向编码为 ACN/SN3D 增益 [W, Y, Z, X]
export function encodeFOA(azimuthDeg, elevationDeg) {
  const az = azimuthDeg * DEG;
  const el = elevationDeg * DEG;
  const ce = Math.cos(el);
  return [1, ce * Math.sin(az), Math.sin(el), ce * Math.cos(az)];
}

// 由头部方位 (yaw, pitch, roll)（度）构造 3x3 矩阵，
// 把世界系声场向量 (X, Y, Z) 变换到头相对系: M = Rx(-roll) * Ry(pitch) * Rz(-yaw)
export function rotationMatrix(yawDeg = 0, pitchDeg = 0, rollDeg = 0) {
  const cy = Math.cos(yawDeg * DEG), sy = Math.sin(yawDeg * DEG);
  const cp = Math.cos(pitchDeg * DEG), sp = Math.sin(pitchDeg * DEG);
  const cr = Math.cos(rollDeg * DEG), sr = Math.sin(rollDeg * DEG);
  const rz = [[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]];
  const ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]];
  const rx = [[1, 0, 0], [0, cr, sr], [0, -sr, cr]];
  return multiply3(multiply3(rx, ry), rz);
}

function multiply3(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

// 用 3x3 矩阵旋转 ACN/SN3D 四通道帧 [W, Y, Z, X]（W 不变）
export function rotateFOA(frame, m) {
  const [w, y, z, x] = frame;
  const x2 = m[0][0] * x + m[0][1] * y + m[0][2] * z;
  const y2 = m[1][0] * x + m[1][1] * y + m[1][2] * z;
  const z2 = m[2][0] * x + m[2][1] * y + m[2][2] * z;
  return [w, y2, z2, x2];
}

// 基本立体声解码: 虚拟心形话筒置于 ±30°，返回 [L, R]
export function decodeStereo(frame) {
  const [w, y, , x] = frame;
  const c = Math.cos(30 * DEG), s = Math.sin(30 * DEG);
  return [0.5 * (w + c * x + s * y), 0.5 * (w + c * x - s * y)];
}

// 把角度规整到 (-180, 180]
export function wrapDeg(deg) {
  let d = deg % 360;
  if (d <= -180) d += 360;
  if (d > 180) d -= 360;
  return d;
}
