'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { createServer, pythonRunner } = require('../src/server.js');

const ROOT = path.resolve(__dirname, '..', '..', '..');

function wavBytes(seconds, rate = 8000) {
  const n = Math.floor(seconds * rate);
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(rate, 24); buf.writeUInt32LE(rate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i += 1) buf.writeInt16LE(Math.round(16000 * Math.sin((2 * Math.PI * 440 * i) / rate)), 44 + i * 2);
  return buf;
}

let tmp;
let server;
let base;
const calls = [];

before(async () => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'nam-api-'));
  const real = pythonRunner(path.join(ROOT, 'services', 'audio'));
  // Real Python for probe/peaks; stubbed Python for the heavy isolate/match/render commands.
  const runPython = async (request) => {
    calls.push(request.command);
    if (request.command === 'isolate') {
      fs.writeFileSync(request.out, wavBytes(0.5));
      return { path: request.out, durationSec: 0.5, sampleRate: 8000 };
    }
    if (request.command === 'match') return { results: [{ name: 'a.nam', distance: 1.5 }, { name: 'b.nam', distance: 2 }] };
    if (request.command === 'render') {
      fs.writeFileSync(request.out, wavBytes(0.2));
      return { path: request.out, sampleRate: 8000 };
    }
    return real(request);
  };
  server = createServer({ dataDir: path.join(tmp, 'data'), modelsDir: path.join(tmp, 'models'), runPython, webDir: path.join(tmp, 'web') });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => {
  server.close();
  fs.rmSync(tmp, { recursive: true, force: true });
});

const upload = (bytes, name = 'song.wav') => fetch(`${base}/api/tracks`, { method: 'POST', headers: { 'x-filename': name }, body: bytes });
const postJson = (url, body) => fetch(base + url, { method: 'POST', body: JSON.stringify(body) });

test('upload, peaks, bad ranges, isolate, guitar, match and render', async () => {
  const up = await upload(wavBytes(2));
  assert.strictEqual(up.status, 200);
  const track = await up.json();
  assert.strictEqual(track.name, 'song.wav');
  assert.strictEqual(track.sampleRate, 8000);
  assert.ok(Math.abs(track.durationSec - 2) < 0.01);
  assert.ok(fs.existsSync(path.join(tmp, 'data', `${track.id}.source`)));
  assert.ok(fs.existsSync(path.join(tmp, 'data', `${track.id}.json`)));

  const peaks = await (await fetch(`${base}/api/tracks/${track.id}/peaks`)).json();
  assert.strictEqual(peaks.peaks.length, 1000);
  assert.ok(peaks.peaks.every((p) => p >= 0 && p <= 1));
  assert.ok(peaks.peaks[10] > 0.3);

  for (const body of [{ startSec: 1, endSec: 0.5 }, { startSec: 1, endSec: 1 }, { startSec: -1, endSec: 1 },
    { startSec: 0, endSec: 99 }, { startSec: 'a', endSec: 1 }, {}]) {
    const bad = await postJson(`/api/tracks/${track.id}/isolate`, body);
    assert.strictEqual(bad.status, 400, JSON.stringify(body));
    assert.ok((await bad.json()).error);
  }
  assert.ok(!calls.includes('isolate'), 'bad ranges never reach Python');

  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/guitar`)).status, 404);
  assert.strictEqual((await postJson(`/api/tracks/${track.id}/match`, {})).status, 409);

  const iso = await postJson(`/api/tracks/${track.id}/isolate`, { startSec: 0.5, endSec: 1 });
  assert.strictEqual(iso.status, 200);
  assert.deepStrictEqual(await iso.json(), { id: track.id, startSec: 0.5, endSec: 1, durationSec: 0.5 });

  const guitar = await fetch(`${base}/api/tracks/${track.id}/guitar`);
  assert.strictEqual(guitar.headers.get('content-type'), 'audio/wav');
  assert.strictEqual(Buffer.from(await guitar.arrayBuffer()).toString('ascii', 0, 4), 'RIFF');

  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam`)).status, 400, 'before match');
  const match = await (await postJson(`/api/tracks/${track.id}/match`, {})).json();
  assert.deepStrictEqual(match, { results: [{ name: 'a.nam', distance: 1.5 }, { name: 'b.nam', distance: 2 }] });

  const rendered = await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam`);
  assert.strictEqual(rendered.status, 200);
  assert.strictEqual(rendered.headers.get('content-type'), 'audio/wav');
  for (const bad of ['..%2Fa.nam', 'c.nam', 'sub%5Ca.nam', '%2e%2e%2fa.nam']) {
    const r = await fetch(`${base}/api/tracks/${track.id}/rendered/${bad}`);
    assert.strictEqual(r.status, 400, bad);
  }
});

test('upload validation', async () => {
  const noFile = await upload(Buffer.alloc(0));
  assert.strictEqual(noFile.status, 400);
  const badExt = await upload(wavBytes(1), 'notes.txt');
  assert.strictEqual(badExt.status, 400);
  assert.ok((await badExt.json()).error);
  const noName = await fetch(`${base}/api/tracks`, { method: 'POST', body: wavBytes(1) });
  assert.strictEqual(noName.status, 400);
  const notAudio = await upload(Buffer.from('this is not a wav file at all'), 'fake.wav');
  assert.strictEqual(notAudio.status, 400);
});

test('unknown track and routes return JSON 404', async () => {
  for (const p of ['/api/tracks/nope/peaks', '/api/nothing']) {
    const r = await fetch(base + p);
    assert.strictEqual(r.status, 404);
    assert.ok((await r.json()).error);
  }
});

test('demucs_not_installed surfaces as HTTP 503', async () => {
  const local = createServer({
    dataDir: path.join(tmp, 'data2'),
    runPython: async (request) => (request.command === 'probe'
      ? { durationSec: 5, sampleRate: 44100 }
      : { error: 'demucs_not_installed' }),
  });
  await new Promise((resolve) => local.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${local.address().port}`;
  try {
    const up = await (await fetch(`${url}/api/tracks`, { method: 'POST', headers: { 'x-filename': 'a.mp3' }, body: Buffer.from('x') })).json();
    const r = await fetch(`${url}/api/tracks/${up.id}/isolate`, { method: 'POST', body: JSON.stringify({ startSec: 0, endSec: 1 }) });
    assert.strictEqual(r.status, 503);
    assert.match((await r.json()).error, /demucs/i);
  } finally {
    local.close();
  }
});
