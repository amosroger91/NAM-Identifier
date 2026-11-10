'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { createServer, pythonRunner } = require('../src/server.js');

const ROOT = path.resolve(__dirname, '..', '..', '..');

function wavBytes(seconds, rate = 8000, channels = 1) {
  const n = Math.floor(seconds * rate);
  const frameBytes = channels * 2;
  const buf = Buffer.alloc(44 + n * frameBytes);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * frameBytes, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(rate, 24); buf.writeUInt32LE(rate * frameBytes, 28); buf.writeUInt16LE(frameBytes, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * frameBytes, 40);
  for (let i = 0; i < n; i += 1) buf.writeInt16LE(Math.round(16000 * Math.sin((2 * Math.PI * 440 * i) / rate)), 44 + i * frameBytes);
  return buf;
}

let tmp;
let server;
let base;
const calls = [];
const captureBytes = Buffer.from('test capture contents');
const tone = {
  gain: { label: 'crunch', name: 'Crunch', score: 0.4, confidence: 0.7, method: 'harmonics' },
  adjectives: ['mid-heavy'],
  features: { brightnessDb: 1.5, crestDb: 11, thd: 0.12 },
  clap: { available: false, model: null, matches: [], reason: 'off' },
};
const toneQuery = { ...tone, version: 2, vector: [1], profile: [2], _embedding: null };
const searches = [];

before(async () => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'nam-api-'));
  const real = pythonRunner(path.join(ROOT, 'services', 'audio'));
  // Real Python for probe/peaks; stubbed Python for the heavy isolate/match/render commands.
  const runPython = async (request) => {
    calls.push(request.command);
    if (request.command === 'isolate') {
      fs.writeFileSync(request.out, wavBytes(0.5));
      assert.equal(request.describeTone, true);
      return { path: request.out, durationSec: 0.5, sampleRate: 8000,
        toneQuery: request.startSec === 0 ? null : toneQuery };
    }
    if (request.command === 'match') return {
      results: [{ name: 'a.nam', distance: 1.5 }, { name: 'b.nam', distance: 2 }],
      drive: { label: 'clean', score: 0.15, unmeasuredCaptures: 0 },
    };
    if (request.command === 'describe') return toneQuery;
    if (request.command === 'tone-search') {
      searches.push(request.toneQuery);
      return {
        query: tone,
        ...(request.includeQuery ? { toneQuery } : {}),
        results: [{
          name: 'a.nam', similarity: 91,
          gain: { label: 'edge', name: 'Edge of Breakup', score: 0.2, confidence: 0.8, method: 'harmonics' },
          adjectives: ['bright'],
        }],
        searched: 1, ranked: 1, skipped: [], cached: 0, built: 1,
        clap: { available: false, model: null, matches: [], reason: 'off' },
      };
    }
    if (request.command === 'audition-source') return { path: path.join(tmp, 'guitar.wav') };
    if (request.command === 'audition') {
      assert.strictEqual(path.basename(request.capture), 'a.nam');
      assert.strictEqual(request.source, path.join(tmp, 'guitar.wav'));
      assert.strictEqual(request.gainDb, undefined);
      fs.writeFileSync(request.out, wavBytes(0.2, 8000, request.compare ? 2 : 1));
      return { path: request.out, sampleRate: 8000 };
    }
    return real(request);
  };
  fs.mkdirSync(path.join(tmp, 'models'));
  fs.writeFileSync(path.join(tmp, 'models', 'a.nam'), captureBytes);
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
  assert.deepStrictEqual(await iso.json(), { id: track.id, startSec: 0.5, endSec: 1, durationSec: 0.5, revision: 1, tone });

  const guitar = await fetch(`${base}/api/tracks/${track.id}/guitar`);
  assert.strictEqual(guitar.headers.get('content-type'), 'audio/wav');
  assert.strictEqual(Buffer.from(await guitar.arrayBuffer()).toString('ascii', 0, 4), 'RIFF');

  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam`)).status, 400, 'before match');
  const match = await (await postJson(`/api/tracks/${track.id}/match`, {})).json();
  assert.deepStrictEqual(match, {
    results: [{ name: 'a.nam', distance: 1.5 }, { name: 'b.nam', distance: 2 }], revision: 1, searched: 2, ranked: 2, skipped: [], bank: null,
    drive: { label: 'clean', score: 0.15, unmeasuredCaptures: 0 },
  });

  const previewUrl = `${base}/api/tracks/${track.id}/rendered/a.nam`;
  const previews = await Promise.all([fetch(previewUrl), fetch(previewUrl)]);
  for (const rendered of previews) {
    assert.strictEqual(rendered.status, 200);
    assert.strictEqual(rendered.headers.get('content-type'), 'audio/wav');
    assert.strictEqual(Buffer.from(await rendered.arrayBuffer()).toString('ascii', 0, 4), 'RIFF');
  }
  assert.strictEqual(calls.filter((cmd) => cmd === 'audition').length, 1);
  const downloadUrl = `${base}/api/tracks/${track.id}/download/a.nam`;
  const download = await fetch(downloadUrl + '?rev=1');
  assert.strictEqual(download.status, 200);
  assert.strictEqual(download.headers.get('content-disposition'), "attachment; filename*=UTF-8''a.nam");
  assert.deepStrictEqual(Buffer.from(await download.arrayBuffer()), captureBytes);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/download/..%2Fa.nam`)).status, 400);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/download/b.nam`)).status, 404);
  // a new isolation is a new immutable revision: everything tied to the old one is rejected as stale
  const iso2 = await postJson(`/api/tracks/${track.id}/isolate`, { startSec: 0, endSec: 1 });
  assert.strictEqual((await iso2.json()).revision, 2);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/guitar?rev=1`)).status, 409);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/guitar?rev=2`)).status, 200);
  assert.strictEqual((await postJson(`/api/tracks/${track.id}/match`, { revision: 1 })).status, 409);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam`)).status, 400, 'matches cleared by new isolation');
  const rematch = await (await postJson(`/api/tracks/${track.id}/match`, { revision: 2 })).json();
  assert.strictEqual(rematch.revision, 2);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam?rev=1`)).status, 409);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam?rev=2`)).status, 200);
  assert.strictEqual((await fetch(downloadUrl + '?rev=1')).status, 409);
  assert.strictEqual(calls.filter((cmd) => cmd === 'audition').length, 1, 'auditions are reused across revisions');
  const compareUrl = previewUrl + '?rev=2&compare=1';
  for (let i = 0; i < 2; i += 1) {
    const response = await fetch(compareUrl);
    assert.strictEqual(response.status, 200);
    assert.strictEqual(Buffer.from(await response.arrayBuffer()).readUInt16LE(22), 2);
  }
  assert.strictEqual(calls.filter((cmd) => cmd === 'audition').length, 2, 'comparison audio has its own cache');
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

test('describe and tone search use the isolated guitar and can be auditioned', async () => {
  const up = await upload(wavBytes(2), 'tone.wav');
  const track = await up.json();
  assert.strictEqual((await postJson(`/api/tracks/${track.id}/describe`, {})).status, 409);
  assert.strictEqual((await postJson(`/api/tracks/${track.id}/tone-search`, {})).status, 409);
  const iso = await postJson(`/api/tracks/${track.id}/isolate`, { startSec: 0.2, endSec: 1 });
  assert.strictEqual(iso.status, 200);
  assert.deepStrictEqual((await iso.json()).tone, tone);
  const described = await (await postJson(`/api/tracks/${track.id}/describe`, { revision: 1 })).json();
  assert.strictEqual(described.revision, 1);
  assert.strictEqual(described.gain.name, 'Crunch');
  assert.deepStrictEqual(described.adjectives, ['mid-heavy']);
  const found = await (await postJson(`/api/tracks/${track.id}/tone-search`, { revision: 1 })).json();
  assert.strictEqual(found.revision, 1);
  assert.strictEqual(found.results[0].name, 'a.nam');
  assert.strictEqual(found.results[0].similarity, 91);
  assert.deepStrictEqual(found.query, tone);
  assert.deepStrictEqual(searches.at(-1), toneQuery);
  assert.ok(!calls.includes('describe'), 'the description endpoint reuses the isolation analysis');
  const rendered = await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam?rev=1`);
  assert.strictEqual(rendered.status, 200);
  assert.strictEqual(Buffer.from(await rendered.arrayBuffer()).toString('ascii', 0, 4), 'RIFF');
  const again = await postJson(`/api/tracks/${track.id}/isolate`, { startSec: 0, endSec: 1 });
  const replacement = await again.json();
  assert.strictEqual(replacement.revision, 2);
  assert.strictEqual(replacement.tone, null);
  assert.strictEqual((await fetch(`${base}/api/tracks/${track.id}/rendered/a.nam?rev=1`)).status, 400);
  assert.strictEqual((await postJson(`/api/tracks/${track.id}/tone-search`, { revision: 1 })).status, 409);
  await postJson(`/api/tracks/${track.id}/tone-search`, { revision: 2 });
  assert.strictEqual(searches.at(-1), null, 'a replacement stem gets its own analysis');
  await postJson(`/api/tracks/${track.id}/tone-search`, { revision: 2 });
  assert.deepStrictEqual(searches.at(-1), toneQuery, 'fallback analysis is reused on subsequent searches');
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
