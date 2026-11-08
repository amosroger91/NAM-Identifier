import React, { useEffect, useRef, useState } from 'react';
import { clampRegion } from './region.js';

async function api(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try {
    data = await res.json();
  } catch (_e) {
    /* non-JSON body */
  }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

const fmt = (s) => (Number.isFinite(s) ? s.toFixed(2) : '–');

function Waveform({ peaks, durationSec, region, onSelect }) {
  const canvasRef = useRef(null);
  const dragRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 140;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#101215';
    ctx.fillRect(0, 0, width, height);
    const top = peaks.reduce((m, p) => Math.max(m, p), 0) || 1; // scale quiet files up for display
    ctx.fillStyle = '#5b86b5';
    peaks.forEach((p, i) => {
      const x = (i / peaks.length) * width;
      const h = Math.max(1, (p / top) * height);
      ctx.fillRect(x, (height - h) / 2, Math.max(1, width / peaks.length), h);
    });
    if (region) {
      const x0 = (region.startSec / durationSec) * width;
      const x1 = (region.endSec / durationSec) * width;
      ctx.fillStyle = 'rgba(224,160,48,0.3)';
      ctx.fillRect(x0, 0, x1 - x0, height);
      ctx.fillStyle = '#e0a030';
      ctx.fillRect(x0, 0, 2, height);
      ctx.fillRect(x1 - 2, 0, 2, height);
    }
  }, [peaks, durationSec, region]);

  const timeAt = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    return Math.min(durationSec, Math.max(0, ratio * durationSec)); // pointer position, not a region clamp
  };

  const onPointerDown = (e) => {
    canvasRef.current.setPointerCapture(e.pointerId);
    dragRef.current = timeAt(e);
    onSelect({ startSec: dragRef.current, endSec: dragRef.current });
  };
  const onPointerMove = (e) => {
    if (dragRef.current === null) return;
    const t = timeAt(e);
    onSelect({ startSec: Math.min(dragRef.current, t), endSec: Math.max(dragRef.current, t) });
  };
  const onPointerUp = () => {
    dragRef.current = null;
  };
  const onKeyDown = (e) => {
    // Keyboard selection: arrows move the start, shift+arrows move the end, 1 s steps.
    const step = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0;
    if (!step) return;
    e.preventDefault();
    const cur = region || { startSec: 0, endSec: Math.min(durationSec, 5) };
    const next = { ...cur };
    if (e.shiftKey) next.endSec = Math.min(durationSec, Math.max(0, cur.endSec + step));
    else next.startSec = Math.min(durationSec, Math.max(0, cur.startSec + step));
    onSelect(next);
  };

  return (
    <canvas
      ref={canvasRef}
      tabIndex={0}
      role="img"
      aria-label="Waveform. Drag to select a range, or use arrow keys to move the start and shift plus arrow keys to move the end."
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onKeyDown={onKeyDown}
    />
  );
}

export default function App() {
  const [track, setTrack] = useState(null);
  const [peaks, setPeaks] = useState([]);
  const [region, setRegion] = useState(null);
  const [guitarUrl, setGuitarUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [diUrl, setDiUrl] = useState(null);
  const [diName, setDiName] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const regionError = region ? clampRegion(region.startSec, region.endSec, track?.durationSec ?? 0) : null;
  const canIsolate = Boolean(track && region && !regionError && !busy);

  async function run(label, fn) {
    setBusy(label);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  }

  const onFile = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    run('Uploading…', async () => {
      setTrack(null); setPeaks([]); setRegion(null); setGuitarUrl(null); setResults(null); setDiUrl(null);
      const created = await api('/api/tracks', { method: 'POST', headers: { 'x-filename': file.name }, body: file });
      const data = await api(`/api/tracks/${created.id}/peaks`);
      setTrack(created);
      setPeaks(data.peaks);
    });
  };

  const isolate = () => run('Isolating guitar (this can take a while)…', async () => {
    setResults(null); setDiUrl(null);
    await api(`/api/tracks/${track.id}/isolate`, {
      method: 'POST', body: JSON.stringify({ startSec: region.startSec, endSec: region.endSec }),
    });
    setGuitarUrl(`/api/tracks/${track.id}/guitar?t=${Date.now()}`);
  });

  const match = () => run('Matching captures…', async () => {
    const data = await api(`/api/tracks/${track.id}/match`, { method: 'POST' });
    setResults(data.results);
    setDiUrl(null);
  });

  const playDi = (name) => {
    setDiName(name);
    setDiUrl(`/api/tracks/${track.id}/rendered/${encodeURIComponent(name)}`);
  };

  return (
    <main>
      <h1>NAM Tone Match</h1>
      <p className="sub">Pick a section of a song, isolate the guitar, and find the closest Neural Amp Modeler capture.</p>

      <section>
        <label htmlFor="file">Song (MP3 or WAV)</label>
        <input id="file" type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" onChange={onFile} disabled={Boolean(busy)} />
      </section>

      {track && (
        <section>
          <label>{track.name} · {fmt(track.durationSec)} s</label>
          <Waveform peaks={peaks} durationSec={track.durationSec} region={region} onSelect={setRegion} />
          <div className="range" aria-live="polite">
            {region
              ? <>Selected: {fmt(region.startSec)} s to {fmt(region.endSec)} s ({fmt(region.endSec - region.startSec)} s)</>
              : 'Drag across the waveform to select a range.'}
            {region && regionError && <span className="error"> — {regionError}</span>}
          </div>
          <div className="row">
            <button type="button" onClick={isolate} disabled={!canIsolate}>Isolate guitar</button>
            <button type="button" onClick={match} disabled={!guitarUrl || Boolean(busy)}>Match NAM</button>
          </div>
        </section>
      )}

      {busy && <p className="status" role="status">{busy}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {guitarUrl && (
        <section>
          <label>Isolated guitar</label>
          <audio controls autoPlay src={guitarUrl} />
        </section>
      )}

      {results && (
        <section>
          <label>Closest captures (lower distance is closer)</label>
          {results.length === 0 ? (
            <p className="status">No <code>.nam</code> files found in <code>models/</code>.</p>
          ) : (
            <table>
              <thead><tr><th>Capture</th><th>Distance (dB)</th><th>Clean DI</th></tr></thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={r.name} className={i === 0 ? 'best' : ''}>
                    <td>{r.name}</td>
                    <td className="num">{r.distance.toFixed(2)}</td>
                    <td><button type="button" onClick={() => playDi(r.name)} aria-label={`Play clean DI through ${r.name}`}>Play</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {diUrl && (
            <>
              <p className="status">Clean DI through {diName}</p>
              <audio controls autoPlay src={diUrl} />
            </>
          )}
        </section>
      )}
    </main>
  );
}
