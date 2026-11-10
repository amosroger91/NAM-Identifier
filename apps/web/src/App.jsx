import React, { useEffect, useRef, useState } from 'react';
import { clampRegion } from './region.js';
import { EFFECTS, createRack, defaultEffectState } from './effects.js';
import { EffectsPanel, LiveMic } from './panels.jsx';

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
const DRIVE_LABELS = { clean: 'Clean', dirty: 'Dirty / overdriven', uncertain: 'Uncertain / edge of breakup' };

function toneLine(gain, adjectives) {
  const name = gain?.name || 'Uncertain';
  return adjectives?.length ? `${name} · ${adjectives.join(', ')}` : name;
}

function createAuditionGraph(player) {
  const context = new AudioContext();
  const split = context.createChannelSplitter(2);
  const effect = context.createGain();
  const dry = context.createGain();
  dry.gain.value = 0;
  context.createMediaElementSource(player).connect(split);
  // Both versions of the take share one media element and playhead.
  split.connect(effect, 0);
  split.connect(dry, 1);
  const bus = context.createGain();
  effect.connect(bus);
  dry.connect(bus);
  // Optional worship effects sit on top of whichever version is playing.
  const rack = createRack(context);
  bus.connect(rack.input);
  rack.output.connect(context.destination);
  return { context, effect, dry, rack };
}

function setAuditionEffect(graph, on) {
  const now = graph.context.currentTime;
  for (const [node, level] of [[graph.effect, Number(on)], [graph.dry, Number(!on)]]) {
    node.gain.cancelScheduledValues(now);
    node.gain.setTargetAtTime(level, now, 0.004);
  }
}

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
  const [isolated, setIsolated] = useState(null); // { revision, startSec, endSec } of the stem that is playing
  const [summary, setSummary] = useState(null); // { searched, ranked, skipped } of the last match or tone search
  const [tone, setTone] = useState(null);
  const [resultMode, setResultMode] = useState('match');
  const latest = useRef(0); // bumped by every action; a response from an older action is ignored
  const [results, setResults] = useState(null);
  const [preview, setPreview] = useState(null);
  const [effectOn, setEffectOn] = useState(true);
  const [fx, setFx] = useState(defaultEffectState);
  const [fxAvailable, setFxAvailable] = useState({ swell: true, shimmer: true });
  const stemPlayer = useRef(null);
  const previewPlayer = useRef(null);
  const auditionGraph = useRef(null);
  const playback = useRef(0);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const regionError = region ? clampRegion(region.startSec, region.endSec, track?.durationSec ?? 0) : null;
  const canIsolate = Boolean(track && region && !regionError && !busy);
  const resultUrl = (action, name, revision) => `/api/tracks/${track.id}/${action}/${encodeURIComponent(name)}?rev=${revision}`;
  const comparisonUrl = (name, revision) => `${resultUrl('rendered', name, revision)}&compare=1`;
  const first = results?.[0];
  const firstPreviewUrl = first ? comparisonUrl(first.name, first.revision) : null;

  useEffect(() => {
    const player = previewPlayer.current;
    if (player && firstPreviewUrl) player.src = firstPreviewUrl;
    return () => {
      playback.current += 1;
      if (auditionGraph.current) {
        auditionGraph.current.context.close();
        auditionGraph.current = null;
      }
    };
  }, [firstPreviewUrl]);

  async function run(label, fn) {
    const ticket = ++latest.current;
    const current = () => ticket === latest.current;
    setBusy(label);
    setError('');
    try {
      await fn(current);
    } catch (e) {
      if (current()) setError(e.message);
    } finally {
      if (current()) setBusy('');
    }
  }

  const regionChanged = Boolean(isolated && region
    && (Math.abs(isolated.startSec - region.startSec) > 1e-6 || Math.abs(isolated.endSec - region.endSec) > 1e-6));

  const onFile = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    run('Uploading…', async (current) => {
      setTrack(null); setPeaks([]); setRegion(null); setGuitarUrl(null); setIsolated(null);
      setResults(null); setSummary(null); setTone(null); setResultMode('match'); setPreview(null);
      const created = await api('/api/tracks', { method: 'POST', headers: { 'x-filename': file.name }, body: file });
      const data = await api(`/api/tracks/${created.id}/peaks`);
      if (!current()) return;
      setTrack(created);
      setPeaks(data.peaks);
    });
  };

  const isolate = () => run('Isolating guitar (this can take a while)…', async (current) => {
    setResults(null); setSummary(null); setTone(null); setPreview(null); setGuitarUrl(null); setIsolated(null);
    const asked = { startSec: region.startSec, endSec: region.endSec };
    const done = await api(`/api/tracks/${track.id}/isolate`, { method: 'POST', body: JSON.stringify(asked) });
    if (!current()) return;
    setIsolated({ revision: done.revision, ...asked });
    setGuitarUrl(`/api/tracks/${track.id}/guitar?rev=${done.revision}`);
    setTone(done.tone ?? null);
  });

  const match = () => run('Matching captures…', async (current) => {
    playback.current += 1;
    previewPlayer.current?.pause();
    setPreview(null);
    const data = await api(`/api/tracks/${track.id}/match`, {
      method: 'POST', body: JSON.stringify({ revision: isolated.revision }),
    });
    if (!current()) return;
    setResultMode('match');
    setResults(data.results.map((r) => ({ ...r, revision: data.revision })));
    setSummary({
      searched: data.searched, ranked: data.ranked, skipped: data.skipped, bank: data.bank, drive: data.drive, mode: 'match',
    });
  });

  const describeTone = () => run('Describing tone…', async (current) => {
    const data = await api(`/api/tracks/${track.id}/describe`, {
      method: 'POST', body: JSON.stringify({ revision: isolated.revision }),
    });
    if (!current()) return;
    setTone(data);
  });

  const searchTone = () => run('Searching tones…', async (current) => {
    playback.current += 1;
    previewPlayer.current?.pause();
    setPreview(null);
    const data = await api(`/api/tracks/${track.id}/tone-search`, {
      method: 'POST', body: JSON.stringify({ revision: isolated.revision }),
    });
    if (!current()) return;
    setResultMode('tone');
    setTone(data.query);
    setResults(data.results.map((r) => ({ ...r, revision: data.revision })));
    setSummary({
      searched: data.searched, ranked: data.ranked, skipped: data.skipped,
      clap: data.clap, cached: data.cached, built: data.built, mode: 'tone',
    });
  });

  const playCapture = async (name, revision) => {
    const player = previewPlayer.current;
    const url = comparisonUrl(name, revision);
    const ticket = ++playback.current;
    setPreview({ name, loading: true });
    setEffectOn(true);
    stemPlayer.current?.pause();
    if (player.getAttribute('src') !== url || player.error) player.src = url;
    else player.currentTime = 0;
    try {
      const graph = auditionGraph.current ??= createAuditionGraph(player);
      setAuditionEffect(graph, true);
      for (const [id, patch] of Object.entries(fx)) graph.rack.set(id, patch);
      graph.rack.ready.then(() => setFxAvailable(graph.rack.available()));
      await Promise.all([graph.context.resume(), player.play()]);
      if (ticket === playback.current) setPreview({ name });
    } catch (err) {
      if (ticket === playback.current) {
        setPreview({ name, error: err.name === 'NotAllowedError'
          ? 'Press play in the audio controls to start the audition.'
          : 'Could not play this capture. Try Play again.' });
      }
    }
  };

  const changeFx = (id, patch) => {
    setFx((current) => ({ ...current, [id]: { ...current[id], ...patch } }));
    auditionGraph.current?.rack.set(id, patch);
  };

  const toggleEffect = () => {
    if (!auditionGraph.current) return;
    setAuditionEffect(auditionGraph.current, !effectOn);
    setEffectOn(!effectOn);
  };

  return (
    <main>
      <h1>NAM Tone Match</h1>
      <p className="sub">Pick a section of a song, isolate the guitar, and find the closest Neural Amp Modeler capture. Describe the stem, or search the library by how the tone sounds.</p>

      <section>
        <label htmlFor="file">Song (MP3 or WAV)</label>
        <input id="file" type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" onChange={onFile} disabled={Boolean(busy)} />
      </section>

      <section>
        <details className="fx">
          <summary>Worship effects and live mic ({EFFECTS.filter(({ id }) => fx[id].on).length} effects on)</summary>
          <p className="status">These apply to the audition playback and to the live mic.</p>
          <EffectsPanel fx={fx} available={fxAvailable} onChange={changeFx} />
          <label className="fx-section">Live mic</label>
          <LiveMic fx={fx} onAvailable={setFxAvailable} />
        </details>
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
            <button type="button" onClick={match} disabled={!isolated || regionChanged || Boolean(busy)}>Match NAM</button>
            {isolated && !tone && (
              <button type="button" onClick={describeTone} disabled={regionChanged || Boolean(busy)}>Describe tone</button>
            )}
            <button type="button" onClick={searchTone} disabled={!isolated || regionChanged || Boolean(busy)}>Search by tone</button>
          </div>
          {regionChanged && (
            <p className="status" role="status">The selection changed since the guitar was isolated. Isolate again to match this range.</p>
          )}
        </section>
      )}

      {busy && <p className="status" role="status">{busy}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {guitarUrl && (
        <section>
          <label>Isolated guitar</label>
          <audio ref={stemPlayer} controls autoPlay src={guitarUrl} onPlay={() => previewPlayer.current?.pause()} />
        </section>
      )}

      {tone?.gain && !regionChanged && (
        <section>
          <label>Isolated tone</label>
          <p><strong>{tone.gain.name}</strong></p>
          {tone.adjectives?.length > 0 && (
            <div className="tags">{tone.adjectives.map((word) => <span key={word}>{word}</span>)}</div>
          )}
          <p className="status">
            {tone.clap?.available ? `CLAP embeddings (${tone.clap.model})` : 'Signal measurements'}
            {tone.features && (tone.features.tiltDb != null
              ? ` · treble tilt ${fmt(tone.features.tiltDb)} dB/oct`
              : ` · brightness ${fmt(tone.features.brightnessDb)} dB`)}
            {tone.features && ` · crest ${fmt(tone.features.crestDb)} dB`}
            {tone.features?.thd != null && ` · harmonic distortion ${tone.features.thd}`}
          </p>
          {tone.clap?.matches?.length > 0 && (
            <p className="status">
              Closest descriptions: {tone.clap.matches.map((item) => item.text).join(', ')}
            </p>
          )}
        </section>
      )}

      {results && (
        <section>
          <label>{resultMode === 'tone' ? 'Closest tones' : 'Closest captures (lower distance is closer)'}</label>
          {summary?.mode === 'tone' && (
            <p className="status">
              Scores use the isolated tone's measured traits, drive, and spectrum against a shared phrase rendered through each capture.
              {summary.clap?.available ? ' CLAP embeddings are included.' : ' This search uses signal measurements.'}
              {summary.built > 0 ? ` Built ${summary.built} new tone cards.` : ''}
            </p>
          )}
          {summary?.mode === 'tone' && first?.scoreBreakdown && (
            <p className="status">
              Top result score · Tone traits {first.scoreBreakdown.toneTraits} · Drive {first.scoreBreakdown.drive}
              {' · '}Spectrum {first.scoreBreakdown.spectrum}
              {first.scoreBreakdown.embedding != null && ` · CLAP ${first.scoreBreakdown.embedding}`}
            </p>
          )}
          {summary?.drive && (
            <p className="status">
              Guitar tone estimate <strong>{DRIVE_LABELS[summary.drive.label]}</strong>
              {summary.drive.unmeasuredCaptures > 0 && ` · Drive unavailable for ${summary.drive.unmeasuredCaptures} captures`}
            </p>
          )}
          {summary && (
            <p className="status">
              Searched {summary.searched} captures; {summary.ranked} ranked
              {summary.skipped.length > 0 && `, ${summary.skipped.length} skipped (could not be rendered)`}.
              {summary.skipped.length > 0 && (
                <details><summary>Skipped captures</summary>
                  <ul>{summary.skipped.map((k) => <li key={k.name}>{k.name}: {k.error}</li>)}</ul>
                </details>
              )}
            </p>
          )}
          {summary && summary.bank && summary.bank.recordedPhrases === 0 && (
            <p className="error" role="note">
              No recorded-guitar bank is configured (set NAM_DI_BANK to a folder of dry guitar notes), so this ranking
              uses synthetic phrases only and is much less reliable on real recordings.
            </p>
          )}
          {preview && (
            <>
              <p className={preview.error ? 'error' : 'status'} role="status">
                {preview.error || (preview.loading ? `Preparing ${preview.name}…` : `Auditioning ${preview.name}`)}
              </p>
              <div className="row">
                <button type="button" aria-pressed={effectOn} disabled={preview.loading || Boolean(preview.error)}
                  onClick={toggleEffect}>NAM effect</button>
                <span>{effectOn ? 'On · through NAM' : 'Off · dry guitar'}</span>
              </div>
            </>
          )}
          {first && <audio key={firstPreviewUrl} ref={previewPlayer} loop hidden={!preview} controls preload="auto"
            aria-label="Capture audition" onPlay={() => {
              auditionGraph.current?.context.resume();
              stemPlayer.current?.pause();
            }} />}
          {results.length === 0 ? (
            <p className="status">No <code>.nam</code> files found in <code>models/</code>.</p>
          ) : (
            <table>
              <thead><tr>
                <th>Capture</th>
                {resultMode === 'tone' ? <><th>Tone</th><th>Similarity</th></> : <th>Match distance</th>}
                <th>Audition</th><th>File</th>
              </tr></thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={r.name} className={i === 0 ? 'best' : ''}>
                    <td>{r.name}</td>
                    {resultMode === 'tone' ? (
                      <>
                        <td>{toneLine(r.gain, r.adjectives)}</td>
                        <td className="num">{r.similarity}%</td>
                      </>
                    ) : (
                      <td className="num">{Number(r.distance).toFixed(2)}</td>
                    )}
                    <td><button type="button" onClick={() => playCapture(r.name, r.revision)} aria-label={`Play ${r.name}`}>Play</button></td>
                    <td><a className="download" href={resultUrl('download', r.name, r.revision)} download={r.name}
                      aria-label={`Download ${r.name}`}>Download NAM</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}
    </main>
  );
}
