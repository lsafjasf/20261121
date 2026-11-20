import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseSrt } from './srt.mjs';
import { Timeline } from './timeline.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 8080);

const cuesA = parseSrt(fs.readFileSync(path.join(root, 'subtitles', 'track-a.srt'), 'utf8'));
const cuesB = parseSrt(fs.readFileSync(path.join(root, 'subtitles', 'track-b.srt'), 'utf8'));
const timeline = new Timeline(cuesA, cuesB);
const sseClients = new Set();

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.srt': 'application/x-subrip; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

function sendJson(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(data) });
  res.end(data);
}

function serveFile(res, relative) {
  const filePath = path.normalize(path.join(root, relative));
  if (!filePath.startsWith(root)) return sendJson(res, 403, { error: 'forbidden' });
  fs.readFile(filePath, (err, data) => {
    if (err) return sendJson(res, 404, { error: 'not found' });
    res.writeHead(200, { 'Content-Type': mime[path.extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  });
}

function broadcast() {
  const payload = `data: ${JSON.stringify(timeline.snapshot())}\n\n`;
  for (const res of sseClients) res.write(payload);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 64 * 1024) req.destroy();
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const route = `${req.method} ${url.pathname}`;

  if (route === 'GET /' || route === 'GET /director') {
    return serveFile(res, 'public/director.html');
  }
  if (route === 'GET /viewer') return serveFile(res, 'public/viewer.html');
  if (route.startsWith('GET /public/')) return serveFile(res, url.pathname);
  if (route.startsWith('GET /subtitles/')) return serveFile(res, url.pathname.slice(1));

  if (route === 'GET /api/time') {
    const t0 = Date.now();
    return sendJson(res, 200, { clientMs: Number(url.searchParams.get('t')) || null, serverMs: t0 });
  }
  if (route === 'GET /api/state') return sendJson(res, 200, timeline.snapshot());

  if (route === 'GET /api/events') {
    const clientId = url.searchParams.get('cid') || `c-${sseClients.size + 1}-${Date.now()}`;
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    res.write(`retry: 1000\n\ndata: ${JSON.stringify(timeline.snapshot())}\n\n`);
    sseClients.add(res);
    res.clientId = clientId;
    const ping = setInterval(() => {
      res.write(`event: heartbeat\ndata: ${JSON.stringify({ serverMs: Date.now() })}\n\n`);
    }, 1000);
    req.on('close', () => {
      clearInterval(ping);
      sseClients.delete(res);
    });
    return;
  }

  if (route === 'POST /api/disconnect' && process.env.ENABLE_FAULT_API === '1') {
    const target = url.searchParams.get('cid');
    let closed = 0;
    for (const client of sseClients) {
      if (!target || client.clientId === target) {
        client.destroy();
        closed += 1;
      }
    }
    return sendJson(res, 200, { closed });
  }

  if (route === 'POST /api/control') {
    let body;
    try {
      body = await readBody(req);
    } catch {
      return sendJson(res, 400, { error: 'invalid json' });
    }
    const ok = timeline.apply(body.action, body.payload || {});
    if (!ok) return sendJson(res, 400, { error: 'unknown action' });
    broadcast();
    return sendJson(res, 200, timeline.snapshot());
  }

  sendJson(res, 404, { error: 'not found' });
});

server.listen(port, () => {
  console.log(`director page:  http://localhost:${port}/director`);
  console.log(`viewer page:    http://localhost:${port}/viewer`);
});
