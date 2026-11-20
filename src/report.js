'use strict';

function escapeJson(value) {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

function buildReportHtml(metrics) {
  const data = escapeJson(metrics);
  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>滚动小字量化取舍实验</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #101218; color: #e8ecf4; }
header, main { padding: 24px; max-width: 1440px; margin: auto; }
h1 { margin: 0 0 8px; font-size: 26px; }
.muted { color: #a9b3c7; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 18px 0; }
.card { background: #191d27; border: 1px solid #2b3242; border-radius: 12px; padding: 14px; }
.card h2 { margin: 0 0 10px; font-size: 15px; color: #bfd0ee; }
.metric { display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; border-bottom: 1px solid #242a37; }
.metric:last-child { border-bottom: 0; }
.viewer { background: #151923; border: 1px solid #2b3242; border-radius: 14px; padding: 16px; margin: 18px 0; }
.controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
input[type="range"] { width: min(560px, 70vw); }
img { width: 100%; image-rendering: pixelated; border-radius: 8px; background: #090b10; }
.labels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 8px; font-size: 12px; color: #c8d2e5; }
.labels span { background: #202637; padding: 6px; border-radius: 6px; text-align: center; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #151923; border-radius: 12px; overflow: hidden; }
th, td { padding: 8px 10px; border-bottom: 1px solid #262d3c; text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: #aebbd2; background: #1d2430; position: sticky; top: 0; }
.table-wrap { max-height: 420px; overflow: auto; border: 1px solid #2b3242; border-radius: 12px; }
.bad { color: #ffd166; font-weight: 700; }
.good { color: #7ee2a8; }
button { background: #315efb; color: white; border: 0; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
code { color: #9ad7ff; }
</style>
</head>
<body>
<header>
  <h1>滚动小字：整帧统一量化 vs 文字区域优先分配</h1>
  <div class="muted">同一测试片与原始字形掩码；逐帧匹配相近字节预算。左起：原始片、统一量化、ROI 分配、差异图。黄色为统一量化断裂，绿色为 ROI 断裂，紫色为仍可追踪的文字，红色为背景误差。</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="viewer">
    <div class="controls">
      <strong>逐帧翻看</strong>
      <input id="frameRange" type="range" min="0" max="${metrics.config.frames - 1}" value="0">
      <button id="prev">上一帧</button>
      <button id="next">下一帧</button>
      <span id="frameTitle"></span>
    </div>
    <img id="frameImage" alt="逐帧四宫格差异图">
    <div class="labels">
      <span>原始片 + 字形掩码</span>
      <span>整帧统一量化</span>
      <span>文字区域优先</span>
      <span>文字/背景损伤图</span>
    </div>
    <div id="frameMetrics" style="margin-top:14px"></div>
  </section>
  <h2>文字重合处连通笔画</h2>
  <div class="table-wrap"><table id="strokeTable"></table></div>
</main>
<script>
const metrics = ${data};
const fmt = (value, digits = 2) => Number(value).toFixed(digits);
const frameName = index => 'frames/frame_' + String(index).padStart(3, '0') + '.png';
function metricRow(label, value) {
  return '<div class="metric"><span>' + label + '</span><strong>' + value + '</strong></div>';
}
function renderCards() {
  const s = metrics.summary;
  document.getElementById('cards').innerHTML = ['uniform', 'roi'].map(mode => {
    const title = mode === 'uniform' ? '整帧统一量化' : '文字区域优先分配';
    return '<div class="card"><h2>' + title + '</h2>' +
      metricRow('总字节', s[mode === 'uniform' ? 'uniformBytes' : 'roiBytes'].toLocaleString()) +
      metricRow('平均帧字节', fmt(s[mode].avgBytes, 1)) +
      metricRow('文字 PSNR', fmt(s[mode].avgTextPsnr, 2) + ' dB') +
      metricRow('背景 PSNR', fmt(s[mode].avgBackgroundPsnr, 2) + ' dB') +
      metricRow('断裂连通笔画次数', s[mode].totalBrokenStrokes) +
      metricRow('最久消失时长', fmt(s[mode].longestMissingSeconds, 3) + ' 秒') +
      metricRow('背景最差帧', '<button onclick="document.getElementById(\\'frameRange\\').value=' + s[mode].worstBackgroundFrame + ';renderFrame(' + s[mode].worstBackgroundFrame + ')">第 ' + s[mode].worstBackgroundFrame + ' 帧</button>') +
      '</div>';
  }).join('') + '<div class="card"><h2>预算约束</h2>' +
    metricRow('ROI / 统一字节比', fmt(s.byteRatio, 4)) +
    metricRow('字节差', s.byteDelta) +
    metricRow('测试帧数', metrics.config.frames) +
    metricRow('帧率', metrics.config.fps + ' fps') +
    metricRow('ROI 文字 Q', metrics.config.textQ) +
    '</div>';
}
function renderFrame(index) {
  const frame = metrics.frames[index];
  document.getElementById('frameImage').src = frameName(index);
  document.getElementById('frameTitle').textContent = '第 ' + index + ' 帧 / ' + fmt(frame.timeSeconds, 3) + ' 秒';
  document.getElementById('frameMetrics').innerHTML = '<div class="cards">' + ['uniform', 'roi'].map(mode => {
    const item = frame[mode];
    return '<div class="card"><h2>' + (mode === 'uniform' ? '统一量化' : 'ROI 分配') + '</h2>' +
      metricRow('字节 / 目标', item.bytes + ' / ' + frame.targetBytes) +
      metricRow('Q', mode === 'uniform' ? item.q : '文字 ' + item.textQ + '，背景 ' + item.backgroundQ) +
      metricRow('断裂笔画', item.brokenStrokes) +
      metricRow('丢失文字像素', item.lostTextPixels) +
      metricRow('文字 PSNR', fmt(item.textPsnr, 2) + ' dB') +
      metricRow('背景 PSNR', fmt(item.backgroundPsnr, 2) + ' dB') +
      '</div>';
  }).join('') + '</div>';
  const roiById = new Map(frame.roi.strokes.map(stroke => [stroke.id, stroke]));
  const rows = frame.uniform.strokes.map((stroke) => {
    const other = roiById.get(stroke.id) || { broken: true, longestMissingFrames: 0, continuity: 0, lost: 1 };
    return '<tr><td>' + stroke.id + '</td><td>(' + stroke.x + ',' + stroke.y + ')</td>' +
      '<td class="' + (stroke.broken ? 'bad' : 'good') + '">' + (stroke.broken ? '断裂' : '连续') + '</td>' +
      '<td class="' + (other.broken ? 'bad' : 'good') + '">' + (other.broken ? '断裂' : '连续') + '</td>' +
      '<td>' + fmt(stroke.continuity, 3) + '</td><td>' + fmt(other.continuity, 3) + '</td>' +
      '<td>' + fmt(stroke.lost, 3) + '</td><td>' + fmt(other.lost, 3) + '</td>' +
      '<td>' + fmt(stroke.longestMissingFrames / metrics.config.fps, 3) + ' s</td>' +
      '<td>' + fmt(other.longestMissingFrames / metrics.config.fps, 3) + ' s</td></tr>';
  }).join('');
  document.getElementById('strokeTable').innerHTML = '<thead><tr><th>轨迹 ID</th><th>中心</th><th>统一状态</th><th>ROI 状态</th><th>统一连续度</th><th>ROI 连续度</th><th>统一丢失率</th><th>ROI 丢失率</th><th>统一最久消失</th><th>ROI 最久消失</th></tr></thead><tbody>' + rows + '</tbody>';
}
const range = document.getElementById('frameRange');
range.addEventListener('input', () => renderFrame(Number(range.value)));
document.getElementById('prev').onclick = () => { range.value = Math.max(0, Number(range.value) - 1); renderFrame(Number(range.value)); };
document.getElementById('next').onclick = () => { range.value = Math.min(metrics.config.frames - 1, Number(range.value) + 1); renderFrame(Number(range.value)); };
renderCards();
renderFrame(0);
</script>
</body>
</html>`;
}

module.exports = { buildReportHtml };
