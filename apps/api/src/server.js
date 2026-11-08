'use strict';
// Routes (see SPEC.md): POST /api/tracks, GET /api/tracks/:id/peaks, POST /api/tracks/:id/isolate,
// GET /api/tracks/:id/guitar, POST /api/tracks/:id/match, GET /api/tracks/:id/rendered/:name.
// Anything else is served from apps/web/dist. Default port 3210.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const ROOT = path.resolve(__dirname, '..', '..', '..');
const HOST = '127.0.0.1';
const DEFAULT_PORT = 3210;
const MAX_UPLOAD = 300 * 1024 * 1024;
const AUDIO_EXT = new Set(['.mp3', '.wav']);
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.map': 'application/json', '.ico': 'image/x-icon',
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** Default Python bridge: one JSON request on stdin, one JSON object on stdout. */
function pythonRunner(cwd) {
  return (request) => new Promise((resolve, reject) => {
    const child = spawn('python', ['-m', 'nam_tone_match'], { cwd, windowsHide: true });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', (e) => reject(new Error(`cannot start python: ${e.message}`)));
    child.on('close', () => {
      try {
        resolve(JSON.parse(out.trim().split(/\r?\n/).filter(Boolean).pop() || ''));
      } catch (_e) {
        reject(new Error(`python gave no JSON: ${(err || out).trim().slice(-300)}`));
      }
    });
    child.stdin.end(JSON.stringify(request));
  });
}

function createServer(options = {}) {
  const dataDir = options.dataDir || path.join(ROOT, 'data');
  const modelsDir = options.modelsDir || path.join(ROOT, 'models');
  const webDir = options.webDir || path.join(ROOT, 'apps', 'web', 'dist');
  const runPython = options.runPython || pythonRunner(path.join(ROOT, 'services', 'audio'));
  fs.mkdirSync(dataDir, { recursive: true });

  const sidecarPath = (id) => path.join(dataDir, `${id}.json`);
  const sourcePath = (id) => path.join(dataDir, `${id}.source`);
  const guitarPath = (id) => path.join(dataDir, `${id}.guitar.wav`);

  function loadTrack(id) {
    if (!/^[a-z0-9-]+$/i.test(id)) throw new HttpError(404, 'Track not found');
    try {
      return JSON.parse(fs.readFileSync(sidecarPath(id), 'utf8'));
    } catch (_e) {
      throw new HttpError(404, 'Track not found');
    }
  }
  const saveTrack = (id, meta) => fs.writeFileSync(sidecarPath(id), JSON.stringify(meta));

  async function py(request) {
    let result;
    try {
      result = await runPython(request);
    } catch (e) {
      throw new HttpError(500, e.message);
    }
    if (result && result.error === 'demucs_not_installed') {
      throw new HttpError(503, 'Demucs is not installed. Run: python -m pip install demucs');
    }
    if (result && result.error) throw new HttpError(500, String(result.error));
    return result;
  }

  function readBody(req, limit = MAX_UPLOAD) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      let size = 0;
      req.on('data', (c) => {
        size += c.length;
        if (size > limit) {
          reject(new HttpError(413, 'Upload too large'));
          req.destroy();
          return;
        }
        chunks.push(c);
      });
      req.on('end', () => resolve(Buffer.concat(chunks)));
      req.on('error', reject);
    });
  }

  async function readJson(req) {
    const raw = (await readBody(req, 1024 * 1024)).toString('utf8');
    if (!raw.trim()) return {};
    try {
      const data = JSON.parse(raw);
      if (data === null || typeof data !== 'object' || Array.isArray(data)) throw new Error('x');
      return data;
    } catch (_e) {
      throw new HttpError(400, 'Invalid JSON body');
    }
  }

  function sendJson(res, status, body) {
    const raw = JSON.stringify(body);
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(raw) });
    res.end(raw);
  }

  function sendFile(res, file, type) {
    let stat;
    try {
      stat = fs.statSync(file);
      if (!stat.isFile()) throw new Error('not a file');
    } catch (_e) {
      throw new HttpError(404, 'Not found');
    }
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size });
    fs.createReadStream(file).pipe(res);
  }

  function serveStatic(res, pathname) {
    const rel = decodeURIComponent(pathname === '/' ? '/index.html' : pathname);
    const file = path.normalize(path.join(webDir, rel));
    if (!file.startsWith(webDir + path.sep) || !fs.existsSync(file)) {
      if (rel === '/index.html') {
        throw new HttpError(404, 'Web UI is not built. Run: npm --prefix apps/web run build');
      }
      throw new HttpError(404, 'Not found');
    }
    sendFile(res, file, MIME[path.extname(file).toLowerCase()] || 'application/octet-stream');
  }

  async function handleApi(req, res, parts) {
    const [, resource, id, action, extra] = parts;
    if (resource !== 'tracks') throw new HttpError(404, 'Not found');

    if (req.method === 'POST' && !id) {
      const filename = path.basename(String(req.headers['x-filename'] || '').trim());
      if (!filename) throw new HttpError(400, 'Missing x-filename header');
      if (!AUDIO_EXT.has(path.extname(filename).toLowerCase())) {
        throw new HttpError(400, 'Only .mp3 and .wav files are accepted');
      }
      const body = await readBody(req);
      if (body.length === 0) throw new HttpError(400, 'Missing file');
      const trackId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      fs.writeFileSync(sourcePath(trackId), body);
      let probe;
      try {
        probe = await py({ command: 'probe', path: sourcePath(trackId) });
      } catch (e) {
        fs.rmSync(sourcePath(trackId), { force: true });
        throw new HttpError(e.status === 503 ? 503 : 400, `Could not read audio: ${e.message}`);
      }
      saveTrack(trackId, { name: filename, durationSec: probe.durationSec, sampleRate: probe.sampleRate });
      return sendJson(res, 200, {
        id: trackId, name: filename, durationSec: probe.durationSec, sampleRate: probe.sampleRate,
      });
    }

    if (!id) throw new HttpError(404, 'Not found');
    const meta = loadTrack(id);

    if (req.method === 'GET' && action === 'peaks') {
      const result = await py({ command: 'peaks', path: sourcePath(id), count: 1000 });
      return sendJson(res, 200, { peaks: result.peaks, durationSec: meta.durationSec });
    }

    if (req.method === 'POST' && action === 'isolate') {
      const data = await readJson(req);
      const { startSec, endSec } = data;
      if (typeof startSec !== 'number' || typeof endSec !== 'number'
        || !Number.isFinite(startSec) || !Number.isFinite(endSec)) {
        throw new HttpError(400, 'startSec and endSec must be numbers');
      }
      if (!(startSec < endSec)) throw new HttpError(400, 'startSec must be less than endSec');
      if (startSec < 0 || endSec > meta.durationSec) {
        throw new HttpError(400, `Range must be inside 0..${meta.durationSec} seconds`);
      }
      await py({
        command: 'isolate', path: sourcePath(id), startSec, endSec, out: guitarPath(id),
      });
      delete meta.matches;
      meta.region = { startSec, endSec };
      saveTrack(id, meta);
      return sendJson(res, 200, { id, startSec, endSec, durationSec: endSec - startSec });
    }

    if (req.method === 'GET' && action === 'guitar') {
      if (!fs.existsSync(guitarPath(id))) throw new HttpError(404, 'No isolated guitar yet. Isolate a region first.');
      return sendFile(res, guitarPath(id), 'audio/wav');
    }

    if (req.method === 'POST' && action === 'match') {
      if (!fs.existsSync(guitarPath(id))) throw new HttpError(409, 'Isolate the guitar before matching.');
      const result = await py({
        command: 'match', guitar: guitarPath(id), modelsDir, dataDir,
      });
      const results = Array.isArray(result.results) ? result.results : [];
      meta.matches = results.map((r) => r.name);
      meta.gains = Object.fromEntries(results.map((r) => [r.name, Number(r.inputGainDb) || 0]));
      saveTrack(id, meta);
      return sendJson(res, 200, { results });
    }

    if (req.method === 'GET' && action === 'rendered' && extra) {
      let name;
      try {
        name = decodeURIComponent(extra);
      } catch (_e) {
        throw new HttpError(400, 'Bad capture name');
      }
      if (name.includes('..') || name.includes('/') || name.includes('\\') || name !== path.basename(name)) {
        throw new HttpError(400, 'Bad capture name');
      }
      if (!Array.isArray(meta.matches) || !meta.matches.includes(name)) {
        throw new HttpError(400, 'Unknown capture. Run a match first.');
      }
      const gainDb = (meta.gains && meta.gains[name]) || 0;
      const out = path.join(dataDir, `${id}.rendered.${Buffer.from(name).toString('hex')}.${gainDb}.wav`);
      if (!fs.existsSync(out)) await py({ command: 'render', name, modelsDir, dataDir, out, gainDb });
      return sendFile(res, out, 'audio/wav');
    }

    throw new HttpError(404, 'Not found');
  }

  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || '/', `http://${HOST}`);
      const parts = url.pathname.split('/').filter(Boolean);
      if (parts[0] === 'api') {
        await handleApi(req, res, parts);
      } else if (req.method === 'GET') {
        serveStatic(res, url.pathname);
      } else {
        throw new HttpError(404, 'Not found');
      }
    } catch (e) {
      if (res.headersSent) {
        res.end();
        return;
      }
      sendJson(res, e instanceof HttpError ? e.status : 500, {
        error: e instanceof HttpError ? e.message : 'Server error',
      });
    }
  });
}

if (require.main === module) {
  const port = Number(process.env.PORT) || DEFAULT_PORT;
  createServer().listen(port, HOST, () => {
    console.log(`NAM Tone Match running at http://${HOST}:${port}/`);
  });
}

module.exports = { createServer, pythonRunner, HOST, DEFAULT_PORT };
