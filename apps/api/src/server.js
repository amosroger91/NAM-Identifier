const http = require('node:http');

const PORT = 3210;
const HOST = '127.0.0.1';
const tracks = new Map();

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function send(res, status, body) {
  const raw = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(raw),
  });
  res.end(raw);
}

function wavInfo(buffer) {
  let sampleRate = 48000;
  let channels = 1;
  let bits = 16;
  let dataOffset = 44;
  let dataSize = Math.max(0, buffer.length - 44);
  if (buffer.length >= 44 && buffer.toString('ascii', 0, 4) === 'RIFF') {
    channels = buffer.readUInt16LE(22) || 1;
    sampleRate = buffer.readUInt32LE(24) || 48000;
    bits = buffer.readUInt16LE(34) || 16;
    dataSize = buffer.readUInt32LE(40);
    dataOffset = 44;
  }
  const frameBytes = Math.max(1, (bits / 8) * channels);
  const frames = Math.floor(dataSize / frameBytes);
  return {
    sampleRate,
    channels,
    bits,
    dataOffset,
    dataSize,
    durationSec: sampleRate > 0 ? frames / sampleRate : 0,
  };
}

function peaksFromWav(buffer, info) {
  const count = 1000;
  const peaks = new Array(count).fill(0);
  if (info.bits !== 16) return peaks;
  const frameBytes = Math.max(1, (info.bits / 8) * info.channels);
  const frames = Math.floor(info.dataSize / frameBytes);
  if (frames < 1) return peaks;
  for (let i = 0; i < count; i += 1) {
    const frame = Math.min(frames - 1, Math.floor(((i + 0.5) * frames) / count));
    const offset = info.dataOffset + frame * frameBytes;
    if (offset + 2 > buffer.length) break;
    peaks[i] = Math.min(1, Math.abs(buffer.readInt16LE(offset)) / 32768);
  }
  return peaks;
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    const parts = url.pathname.split('/').filter(Boolean);

    if (req.method === 'POST' && url.pathname === '/api/tracks') {
      const body = await readBody(req);
      const filename = String(req.headers['x-filename'] || 'track.wav');
      const info = wavInfo(body);
      const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      tracks.set(id, { buffer: body, info });
      send(res, 200, {
        id,
        name: filename,
        durationSec: info.durationSec,
        sampleRate: info.sampleRate,
      });
      return;
    }

    if (req.method === 'GET' && parts[0] === 'api' && parts[1] === 'tracks' && parts[3] === 'peaks') {
      const track = tracks.get(parts[2]);
      if (!track) {
        send(res, 404, { error: 'Not Found' });
        return;
      }
      send(res, 200, { peaks: peaksFromWav(track.buffer, track.info) });
      return;
    }

    if (req.method === 'POST' && parts[0] === 'api' && parts[1] === 'tracks' && parts[3] === 'isolate') {
      const raw = await readBody(req);
      let data;
      try {
        data = JSON.parse(raw.toString('utf8') || '{}');
      } catch (_err) {
        send(res, 400, { error: 'Invalid JSON body' });
        return;
      }
      const startSec = Number(data.startSec);
      const endSec = Number(data.endSec);
      if (!(startSec < endSec)) {
        send(res, 400, { error: 'Start time must be before end time.' });
        return;
      }
      send(res, 200, { id: parts[2], startSec, endSec });
      return;
    }

    send(res, 404, { error: 'Not Found' });
  } catch (_err) {
    if (!res.headersSent) send(res, 500, { error: 'Server error' });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Server running at http://${HOST}:${PORT}/`);
});
