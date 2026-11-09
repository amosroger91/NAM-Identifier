'use strict';
// Routes (see SPEC.md): POST /api/tracks, GET /api/tracks/:id/peaks, POST /api/tracks/:id/isolate,
// GET /api/tracks/:id/guitar, POST /api/tracks/:id/match, POST /api/tracks/:id/describe,
// POST /api/tracks/:id/tone-search, GET /api/tracks/:id/{rendered,download}/:name.
// Anything else is served from apps/web/dist. Default port 3210.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createHash, randomUUID } = require('node:crypto');
const { createInterface } = require('node:readline');

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

function auditionRunner(cwd) {
  let worker;
  let nextTicket = 0;
  const start = () => {
    const child = spawn('python', ['-m', 'nam_tone_match', '--audition-worker'], { cwd, windowsHide: true });
    const pending = new Map();
    let diagnostics = '';
    child.stderr.on('data', (chunk) => { diagnostics = (diagnostics + chunk).slice(-800); });
    createInterface({ input: child.stdout }).on('line', (line) => {
      try {
        const { ticket, result } = JSON.parse(line);
        pending.get(ticket)?.resolve(result);
        pending.delete(ticket);
      } catch (_e) {
        // A library may write progress to stdout outside Python's redirect.
      }
    });
    const fail = (err) => {
      if (worker?.child === child) worker = undefined;
      for (const { reject } of pending.values()) reject(err);
      pending.clear();
    };
    child.on('error', fail);
    child.stdin.on('error', fail);
    child.on('close', () => fail(new Error(`audition worker stopped: ${diagnostics.trim()}`)));
    return { child, pending };
  };
  const run = (request) => new Promise((resolve, reject) => {
    worker ??= start();
    const ticket = ++nextTicket;
    worker.pending.set(ticket, { resolve, reject });
    worker.child.stdin.write(JSON.stringify({ ...request, ticket }) + '\n');
  });
  return { run, close: () => worker?.child.stdin.end() };
}

function createServer(options = {}) {
  const dataDir = options.dataDir || path.join(ROOT, 'data');
  const modelsDir = options.modelsDir || process.env.NAM_MODELS_DIR || path.join(ROOT, 'models');
  const webDir = options.webDir || path.join(ROOT, 'apps', 'web', 'dist');
  const audioDir = path.join(ROOT, 'services', 'audio');
  const runPython = options.runPython || pythonRunner(audioDir);
  const previewWorker = options.runPython ? null : auditionRunner(audioDir);
  fs.mkdirSync(dataDir, { recursive: true });
  const auditions = new Map();
  const isolating = new Set();
  let auditionSource;

  const sidecarPath = (id) => path.join(dataDir, `${id}.json`);
  const sourcePath = (id) => path.join(dataDir, `${id}.source`);
  const guitarPath = (id, rev) => path.join(dataDir, `${id}.guitar.${rev}.wav`);
  const staleCheck = (value, current, what) => {
    if (value !== null && value !== undefined && String(value) !== String(current)) {
      throw new HttpError(409, `Stale ${what}: the selected region changed. Isolate again.`);
    }
  };

  function loadTrack(id) {
    if (!/^[a-z0-9-]+$/i.test(id)) throw new HttpError(404, 'Track not found');
    try {
      return JSON.parse(fs.readFileSync(sidecarPath(id), 'utf8'));
    } catch (_e) {
      throw new HttpError(404, 'Track not found');
    }
  }
  const saveTrack = (id, meta) => fs.writeFileSync(sidecarPath(id), JSON.stringify(meta));
  const toneQuery = (meta) => meta.toneQuery?.version === 1 ? meta.toneQuery : null;
  const publicTone = (query) => {
    if (!query) return null;
    const { gain, adjectives, features, clap } = query;
    return { gain, adjectives, features, clap };
  };

  async function py(request) {
    let result;
    try {
      const run = previewWorker && request.command.startsWith('audition') ? previewWorker.run : runPython;
      result = await run(request);
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

  function sendFile(res, file, type, headers = {}) {
    let stat;
    try {
      stat = fs.statSync(file);
      if (!stat.isFile()) throw new Error('not a file');
    } catch (_e) {
      throw new HttpError(404, 'Not found');
    }
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size, ...headers });
    fs.createReadStream(file).pipe(res);
  }

  function prepareAudition() {
    auditionSource ??= py({ command: 'audition-source', dataDir }).catch((err) => {
      auditionSource = undefined;
      throw err;
    });
    return auditionSource;
  }

  async function audition(capture, compare) {
    const [model, { path: source }] = await Promise.all([fs.promises.readFile(capture), prepareAudition()]);
    const version = compare ? 'audition-ab-v1\0' : 'audition-v1\0';
    const key = createHash('sha256').update(version).update(path.basename(source)).update(model).digest('hex');
    const out = path.join(dataDir, `audition-${key}.wav`);
    if (!fs.existsSync(out)) {
      if (!auditions.has(key)) {
        const temporary = `${out}.${randomUUID()}.part`;
        const job = py({ command: 'audition', capture, source, out: temporary, compare })
          .then(() => fs.promises.rename(temporary, out))
          .finally(() => {
            auditions.delete(key);
            fs.rmSync(temporary, { force: true });
          });
        auditions.set(key, job);
      }
      await auditions.get(key);
    }
    return out;
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
      if (isolating.has(id)) throw new HttpError(409, 'An isolation is already in progress.');
      const revision = (meta.revision || 0) + 1;
      isolating.add(id);
      let isolated;
      try {
        isolated = await py({
          command: 'isolate', path: sourcePath(id), startSec, endSec, out: guitarPath(id, revision), describeTone: true,
        });
      } finally {
        isolating.delete(id);
      }
      if (meta.revision) fs.rmSync(guitarPath(id, meta.revision), { force: true });
      meta.revision = revision;           // every isolation is an immutable revision; old ones are invalidated
      meta.matches = null;
      meta.toneMatches = null;
      meta.matchRevision = null;
      meta.toneQuery = isolated?.toneQuery ?? null;
      meta.region = { startSec, endSec };
      saveTrack(id, meta);
      return sendJson(res, 200, {
        id, startSec, endSec, durationSec: endSec - startSec, revision, tone: publicTone(toneQuery(meta)),
      });
    }

    if (req.method === 'GET' && action === 'guitar') {
      const url = new URL(req.url, `http://${HOST}`);
      staleCheck(url.searchParams.get('rev'), meta.revision, 'guitar revision');
      if (!meta.revision || !fs.existsSync(guitarPath(id, meta.revision))) {
        throw new HttpError(404, 'No isolated guitar yet. Isolate a region first.');
      }
      return sendFile(res, guitarPath(id, meta.revision), 'audio/wav');
    }

    if (req.method === 'POST' && action === 'match') {
      const data = await readJson(req);
      staleCheck(data.revision, meta.revision, 'match request');
      if (!meta.revision || !fs.existsSync(guitarPath(id, meta.revision))) {
        throw new HttpError(409, 'Isolate the guitar before matching.');
      }
      const revision = meta.revision;
      const result = await py({
        command: 'match', guitar: guitarPath(id, revision), modelsDir, dataDir,
      });
      const current = loadTrack(id);
      staleCheck(revision, current.revision, 'match');
      const results = Array.isArray(result.results) ? result.results : [];
      current.matches = results.map((r) => r.name);
      current.matchRevision = revision;
      current.gains = Object.fromEntries(results.map((r) => [r.name, Number(r.probeInputTrimDb) || 0]));
      saveTrack(id, current);
      return sendJson(res, 200, {
        results,
        revision,
        searched: result.searched ?? results.length,
        ranked: result.ranked ?? results.length,
        skipped: Array.isArray(result.skipped) ? result.skipped : [],
        bank: result.bank ?? null,
        drive: result.drive ?? null,
      });
    }

    if (req.method === 'POST' && (action === 'describe' || action === 'tone-search')) {
      const data = await readJson(req);
      staleCheck(data.revision, meta.revision, action === 'describe' ? 'describe request' : 'tone search');
      if (!meta.revision || !fs.existsSync(guitarPath(id, meta.revision))) {
        throw new HttpError(409, 'Isolate the guitar first.');
      }
      const revision = meta.revision;
      const guitar = guitarPath(id, revision);
      const cached = toneQuery(meta);
      if (action === 'describe') {
        const query = cached || await py({ command: 'describe', path: guitar, includeQuery: true });
        const current = loadTrack(id);
        staleCheck(revision, current.revision, 'tone description');
        if (!cached) {
          current.toneQuery = query;
          saveTrack(id, current);
        }
        return sendJson(res, 200, { ...publicTone(query), revision });
      }
      const result = await py({ command: 'tone-search', guitar, modelsDir, dataDir, toneQuery: cached, includeQuery: !cached });
      const current = loadTrack(id);
      staleCheck(revision, current.revision, 'tone search');
      const results = Array.isArray(result.results) ? result.results : [];
      current.toneMatches = results.map((r) => r.name);
      current.matchRevision = revision;
      if (!cached && result.toneQuery) current.toneQuery = result.toneQuery;
      saveTrack(id, current);
      return sendJson(res, 200, {
        query: result.query ?? null,
        results,
        revision,
        searched: result.searched ?? results.length,
        ranked: result.ranked ?? results.length,
        skipped: Array.isArray(result.skipped) ? result.skipped : [],
        cached: result.cached ?? 0,
        built: result.built ?? 0,
        clap: result.clap ?? null,
      });
    }

    if (req.method === 'GET' && (action === 'rendered' || action === 'download') && extra) {
      let name;
      try {
        name = decodeURIComponent(extra);
      } catch (_e) {
        throw new HttpError(400, 'Bad capture name');
      }
      if (/[\\/\0]/.test(name) || name !== path.basename(name) || path.extname(name).toLowerCase() !== '.nam') {
        throw new HttpError(400, 'Bad capture name');
      }
      const url = new URL(req.url, `http://${HOST}`);
      const known = [
        ...(Array.isArray(meta.matches) ? meta.matches : []),
        ...(Array.isArray(meta.toneMatches) ? meta.toneMatches : []),
      ];
      if (!known.includes(name)) {
        throw new HttpError(400, 'Unknown capture. Run a match or a tone search first.');
      }
      staleCheck(url.searchParams.get('rev'), meta.matchRevision, 'result revision');
      const capture = path.join(modelsDir, name);
      if (!fs.existsSync(capture)) throw new HttpError(404, 'Capture file is no longer in the library.');
      if (action === 'download') {
        const filename = encodeURIComponent(name).replace(/['()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
        return sendFile(res, capture, 'application/octet-stream', { 'Content-Disposition': `attachment; filename*=UTF-8''${filename}` });
      }
      const out = await audition(capture, url.searchParams.get('compare') === '1');
      staleCheck(meta.matchRevision, loadTrack(id).matchRevision, 'result revision');
      return sendFile(res, out, 'audio/wav');
    }

    throw new HttpError(404, 'Not found');
  }

  const server = http.createServer(async (req, res) => {
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
  if (previewWorker) {
    server.once('listening', () => {
      py({ command: 'audition-warm' }).catch(() => {});
      prepareAudition().catch(() => {});
    });
    server.once('close', previewWorker.close);
  }
  return server;
}

// Build the capture caches (match index, then tone cards) in the background at startup so the first
// Match NAM / Search by tone doesn't have to render every capture. Resumable; set NAM_PREWARM=0 to skip.
function prewarmCaches(log = console.log) {
  if (process.env.NAM_PREWARM === '0') return;
  const os = require('node:os');
  const modelsDir = process.env.NAM_MODELS_DIR || path.join(ROOT, 'models');
  const workers = String(Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2))));
  const script = path.join(ROOT, 'scripts', 'index_library.py');
  const steps = [['match index', []], ['tone cards', ['--tone']]];
  const runStep = (i) => {
    if (i >= steps.length) return log('Capture caches are warm.');
    const [label, extra] = steps[i];
    log(`Warming ${label} in the background (${workers} workers)...`);
    const child = spawn('python', [script, `--models=${modelsDir}`, `--workers=${workers}`, ...extra], {
      cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    });
    try { os.setPriority(child.pid, os.constants.priority.PRIORITY_BELOW_NORMAL); } catch { /* best effort */ }
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { out += d; });
    child.on('error', (e) => log(`Cache warm-up (${label}) could not start: ${e.message}`));
    child.on('close', (code) => {
      log(`Warmed ${label}: ${out.trim().split(/\r?\n/)[0] || `exit ${code}`}`);
      runStep(i + 1);
    });
  };
  runStep(0);
}

if (require.main === module) {
  const port = Number(process.env.PORT) || DEFAULT_PORT;
  createServer().listen(port, HOST, () => {
    console.log(`NAM Tone Match running at http://${HOST}:${port}/`);
    prewarmCaches();
  });
}

module.exports = { createServer, pythonRunner, HOST, DEFAULT_PORT };
