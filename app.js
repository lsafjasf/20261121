import { encodeFOA, rotationMatrix, rotateFOA, decodeStereo, wrapDeg } from './foa.js';

const $ = (id) => document.getElementById(id);
const setStatus = (msg) => { $('status').textContent = msg; };

let ctx = null;
let node = null;
let srcNode = null;
let srcKind = null; // 'test' | 'file'
let lastBuf = null;
const head = { yaw: 0, pitch: 0, roll: 0 };
const source = { az: 0, el: 0, active: false, label: '' };

async function ensureAudio() {
  if (ctx) {
    if (ctx.state === 'suspended') await ctx.resume();
    return;
  }
  ctx = new AudioContext();
  await ctx.audioWorklet.addModule('foa-processor.js');
  node = new AudioWorkletNode(ctx, 'foa-listener', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    channelCount: 4,
    channelCountMode: 'explicit',
    channelInterpretation: 'discrete',
  });
  node.connect(ctx.destination);
  node.port.onmessage = onProcessorMessage;
  setStatus('音频已启动。');
}

function applyHead() {
  if (!node) return;
  const t = ctx.currentTime;
  node.parameters.get('yaw').setTargetAtTime(head.yaw, t, 0.03);
  node.parameters.get('pitch').setTargetAtTime(head.pitch, t, 0.03);
  node.parameters.get('roll').setTargetAtTime(head.roll, t, 0.03);
}

function stopSource() {
  if (srcNode) {
    try { srcNode.stop(); } catch { /* 已停止 */ }
    srcNode.disconnect();
    srcNode = null;
  }
  source.active = false;
}

function playBuffer(buf, kind, label) {
  stopSource();
  srcKind = kind;
  srcNode = ctx.createBufferSource();
  srcNode.buffer = buf;
  srcNode.loop = true;
  srcNode.connect(node);
  srcNode.start();
  lastBuf = buf;
}

// 生成 2 秒循环测试声源（脉冲音 + 噪声），按 ACN/SN3D 编码到 4 通道
function makeTestBuffer(az, el) {
  const sr = ctx.sampleRate;
  const len = sr * 2;
  const g = encodeFOA(az, el);
  const buf = ctx.createBuffer(4, len, sr);
  const mono = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const t = i / sr;
    const pulse = Math.max(0, Math.sin(2 * Math.PI * 2 * t)) ** 4;
    mono[i] = 0.35 * Math.sin(2 * Math.PI * 440 * t) * pulse + 0.06 * (Math.random() * 2 - 1);
  }
  for (let ch = 0; ch < 4; ch++) {
    const d = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) d[i] = mono[i] * g[ch];
  }
  return buf;
}

async function playTestSource(az, el, label) {
  await ensureAudio();
  source.az = az;
  source.el = el;
  playBuffer(makeTestBuffer(az, el), 'test', label);
  source.active = true;
  source.label = label;
  setStatus(`测试源播放中：${label} (方位 ${az}°, 俯仰 ${el}°)。转动头部观察通道响应。`);
}

async function loadFile(file) {
  await ensureAudio();
  const data = await file.arrayBuffer();
  const buf = await ctx.decodeAudioData(data);
  if (buf.numberOfChannels !== 4) {
    setStatus(`需要 4 声道 (ACN: W Y Z X) 文件，当前为 ${buf.numberOfChannels} 声道。`);
    return;
  }
  source.active = false;
  playBuffer(buf, 'file', file.name);
  setStatus(`播放文件：${file.name}（4 声道按 ACN/SN3D 解释）。`);
}

// ---- 表桥与环绕对比 ----
const meterNames = ['W', 'Y', 'Z', 'X', 'L', 'R'];
const meterEls = [];
function buildMeters() {
  const box = $('meters');
  for (let i = 0; i < 6; i++) {
    const div = document.createElement('div');
    div.className = 'meter' + (i >= 4 ? ' out' : '');
    div.innerHTML = `<div>${meterNames[i]}</div><div class="bar"><i></i></div><div class="db">-∞</div>`;
    box.appendChild(div);
    meterEls.push({ bar: div.querySelector('i'), db: div.querySelector('.db') });
  }
}

let orbitResolve = null;
let diffResolve = null;
function onProcessorMessage(e) {
  const msg = e.data;
  if (msg.type === 'rms') {
    for (let i = 0; i < 6; i++) {
      const db = 20 * Math.log10(msg.rms[i] + 1e-9);
      meterEls[i].bar.style.height = `${Math.max(0, Math.min(100, 100 + db))}%`;
      meterEls[i].db.textContent = db <= -80 ? '-∞' : db.toFixed(1);
    }
  } else if (msg.type === 'captured' && orbitResolve) {
    orbitResolve();
    orbitResolve = null;
  } else if (msg.type === 'diff' && diffResolve) {
    diffResolve(msg.value);
    diffResolve = null;
  }
}

// ---- 罗盘与方向指示 ----
const canvas = $('compass');
const g2d = canvas.getContext('2d');

function drawCompass() {
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2, R = Math.min(cx, cy) - 34;
  g2d.clearRect(0, 0, w, h);
  g2d.strokeStyle = '#3a4352';
  g2d.lineWidth = 1.5;
  g2d.beginPath();
  g2d.arc(cx, cy, R, 0, Math.PI * 2);
  g2d.stroke();
  g2d.font = '13px system-ui';
  g2d.textAlign = 'center';
  g2d.textBaseline = 'middle';
  const labels = [[0, '前 0°'], [90, '左 90°'], [180, '后 180°'], [-90, '右 -90°']];
  for (const [az, text] of labels) {
    const a = (-90 - az) * Math.PI / 180;
    g2d.fillStyle = '#7b8798';
    g2d.fillText(text, cx + (R + 20) * Math.cos(a), cy + (R + 20) * Math.sin(a));
  }
  for (let az = 0; az < 360; az += 30) {
    const a = (-90 - az) * Math.PI / 180;
    g2d.beginPath();
    g2d.moveTo(cx + (R - 6) * Math.cos(a), cy + (R - 6) * Math.sin(a));
    g2d.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
    g2d.stroke();
  }
  // 声源（世界固定）：俯仰越大越靠近圆心
  if (source.active) {
    const a = (-90 - source.az) * Math.PI / 180;
    const r = R * Math.cos(source.el * Math.PI / 180);
    const sx = cx + r * Math.cos(a), sy = cy + r * Math.sin(a);
    g2d.fillStyle = '#f6ad55';
    g2d.beginPath();
    g2d.arc(sx, sy, 8, 0, Math.PI * 2);
    g2d.fill();
    g2d.fillStyle = '#f6ad55';
    const bearing = wrapDeg(source.az - head.yaw);
    g2d.fillText(`声源相对方位 ${bearing.toFixed(1)}°`, cx, h - 14);
  }
  // 头部箭头（随 yaw 旋转）
  const ha = (-90 - head.yaw) * Math.PI / 180;
  const hx = Math.cos(ha), hy = Math.sin(ha);
  g2d.strokeStyle = '#63b3ed';
  g2d.fillStyle = '#63b3ed';
  g2d.lineWidth = 3;
  g2d.beginPath();
  g2d.moveTo(cx - hx * 14, cy - hy * 14);
  g2d.lineTo(cx + hx * 30, cy + hy * 30);
  g2d.stroke();
  g2d.beginPath();
  g2d.arc(cx, cy, 14, 0, Math.PI * 2);
  g2d.stroke();
  const px = -hy, py = hx;
  g2d.beginPath();
  g2d.moveTo(cx + hx * 40, cy + hy * 40);
  g2d.lineTo(cx + hx * 24 + px * 8, cy + hy * 24 + py * 8);
  g2d.lineTo(cx + hx * 24 - px * 8, cy + hy * 24 - py * 8);
  g2d.closePath();
  g2d.fill();
  g2d.fillStyle = '#9aa7b8';
  g2d.fillText(`yaw ${head.yaw.toFixed(1)}°  pitch ${head.pitch.toFixed(1)}°  roll ${head.roll.toFixed(1)}°`, cx, 16);
  requestAnimationFrame(drawCompass);
}

// ---- 交互 ----
function syncUI() {
  $('yaw').value = wrapDeg(head.yaw);
  $('pitch').value = head.pitch;
  $('roll').value = wrapDeg(head.roll);
  $('yawVal').textContent = `${wrapDeg(head.yaw).toFixed(1)}°`;
  $('pitchVal').textContent = `${head.pitch.toFixed(1)}°`;
  $('rollVal').textContent = `${wrapDeg(head.roll).toFixed(1)}°`;
}

function setHead(part, value) {
  head[part] = part === 'pitch' ? Math.max(-90, Math.min(90, value)) : value;
  applyHead();
  syncUI();
}

let dragging = null;
canvas.addEventListener('pointerdown', (e) => {
  dragging = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  setHead('yaw', head.yaw - (e.clientX - dragging.x) * 0.4);
  setHead('pitch', head.pitch - (e.clientY - dragging.y) * 0.4);
  dragging = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', () => { dragging = null; });

for (const part of ['yaw', 'pitch', 'roll']) {
  $(part).addEventListener('input', (e) => setHead(part, parseFloat(e.target.value)));
}
$('reset').addEventListener('click', () => {
  head.yaw = head.pitch = head.roll = 0;
  applyHead();
  syncUI();
});

$('srcFront').addEventListener('click', () => playTestSource(0, 0, '正前方'));
$('srcSide').addEventListener('click', () => playTestSource(90, 0, '左侧'));
$('srcTop').addEventListener('click', () => playTestSource(0, 90, '正上方'));
$('srcStop').addEventListener('click', () => { stopSource(); setStatus('声源已停止。'); });
$('file').addEventListener('change', (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

// ---- 环绕一圈验证：yaw 0 -> 360，对比首尾输出 ----
// 使用正前方恒定电平参考源，保证环绕前后两个采样窗内容一致（与循环相位无关）。
$('orbit').addEventListener('click', async () => {
  await ensureAudio();
  $('orbit').disabled = true;
  const savedBuf = lastBuf;
  const savedKind = srcKind;
  const sr = ctx.sampleRate;
  const ref = ctx.createBuffer(4, sr, sr);
  const g = encodeFOA(0, 0);
  for (let ch = 0; ch < 4; ch++) ref.getChannelData(ch).fill(0.3 * g[ch]);
  playBuffer(ref, 'ref', '环绕参考源');
  head.yaw = 0;
  node.parameters.get('yaw').setValueAtTime(0, ctx.currentTime);
  syncUI();
  await new Promise((res) => setTimeout(res, 120));
  setStatus('环绕验证中：记录初始输出…');
  await new Promise((res) => { orbitResolve = res; node.port.postMessage('capture'); });
  const dur = 4000;
  const t0 = performance.now();
  await new Promise((res) => {
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      head.yaw = 360 * k;
      node.parameters.get('yaw').setValueAtTime(head.yaw, ctx.currentTime);
      syncUI();
      if (k < 1) requestAnimationFrame(tick);
      else res();
    };
    requestAnimationFrame(tick);
  });
  setStatus('环绕完成，对比输出…');
  await new Promise((res) => setTimeout(res, 250));
  node.port.postMessage('compare');
  const diff = await new Promise((res) => { diffResolve = res; });
  head.yaw = 0;
  node.parameters.get('yaw').setValueAtTime(0, ctx.currentTime);
  syncUI();
  if (savedBuf && savedKind !== 'ref') {
    playBuffer(savedBuf, savedKind, '');
  } else {
    stopSource();
    lastBuf = null;
  }
  const ok = diff < 1e-4;
  setStatus(`环绕一圈回到初始方位：输出最大偏差 ${diff.toExponential(2)} → ${ok ? '可复现，验证通过' : '偏差过大'}`);
  $('orbit').disabled = false;
});

// ---- 页面内数学自检（与 test.mjs 相同断言） ----
function selfTest() {
  const lines = [];
  const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
  const t = (name, cond) => lines.push(`${cond ? 'PASS' : 'FAIL'}  ${name}`);
  const front = encodeFOA(0, 0), left = encodeFOA(90, 0), top = encodeFOA(0, 90);
  t('前源 X=1', near(front[3], 1));
  t('左源 Y=1', near(left[1], 1));
  t('顶源 Z=1', near(top[2], 1));
  const yawed = rotateFOA(front, rotationMatrix(90, 0, 0));
  t('头左转90° 前源移到右侧(Y=-1)', near(yawed[1], -1));
  const [l1, r1] = decodeStereo(rotateFOA(left, rotationMatrix(0, 0, 0)));
  t('左源解码 L>R', l1 > r1);
  const pitched = rotateFOA(top, rotationMatrix(0, 90, 0));
  t('抬头90° 顶源到正前(X=1)', near(pitched[3], 1));
  const a = decodeStereo(rotateFOA(front, rotationMatrix(0, 0, 0)));
  const b = decodeStereo(rotateFOA(front, rotationMatrix(360, 0, 0)));
  t('环绕360° 输出可复现', near(a[0], b[0]) && near(a[1], b[1]));
  const el = $('selftest');
  el.textContent = lines.join('\n');
  el.style.color = lines.every((l) => l.startsWith('PASS')) ? '#68d391' : '#fc8181';
}

buildMeters();
syncUI();
selfTest();
requestAnimationFrame(drawCompass);
