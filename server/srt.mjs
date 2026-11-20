function toMs(h, m, s, ms) {
  return ((Number(h) * 3600) + (Number(m) * 60) + Number(s)) * 1000
    + Number(ms.padEnd(3, '0'));
}

export function parseSrt(text) {
  const normalized = text.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').trim();
  const blocks = normalized.split(/\n{2,}/);
  const cues = [];
  for (const block of blocks) {
    const lines = block.split('\n').filter((line) => line.trim() !== '');
    if (lines.length < 2) continue;
    let headerIndex = 0;
    if (/^\d+$/.test(lines[0].trim())) headerIndex = 1;
    const match = lines[headerIndex].match(
      /(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})/,
    );
    if (!match) continue;
    const start = toMs(match[1], match[2], match[3], match[4]);
    const end = toMs(match[5], match[6], match[7], match[8]);
    cues.push({ start, end, text: lines.slice(headerIndex + 1).join('\n') });
  }
  return cues.sort((a, b) => a.start - b.start);
}
