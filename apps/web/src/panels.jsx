import React, { useEffect, useRef, useState } from 'react';
import { EFFECTS, createRack } from './effects.js';

export function EffectsPanel({ fx, available, onChange }) {
  return (
    <div className="fx-grid">
      {EFFECTS.map((effect) => {
        const unavailable = effect.needsWorklet && !available[effect.id];
        return (
          <div className="fx-card" key={effect.id}>
            <label className="fx-name">
              <input type="checkbox" checked={fx[effect.id].on} disabled={unavailable}
                onChange={(e) => onChange(effect.id, { on: e.target.checked })} />
              {effect.name}
            </label>
            {effect.hint && <small>{unavailable ? 'Not supported in this browser' : effect.hint}</small>}
            {effect.select && (
              <select aria-label={`${effect.name} type`} value={fx[effect.id][effect.select.key]}
                onChange={(e) => onChange(effect.id, { [effect.select.key]: e.target.value })}>
                {effect.select.options.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            )}
            {effect.params.map((param) => (
              <label className="fx-param" key={param.key}>
                <span>{param.label}</span>
                <input type="range" min={param.min} max={param.max} step={param.step}
                  value={fx[effect.id][param.key]}
                  onChange={(e) => onChange(effect.id, { [param.key]: Number(e.target.value) })} />
              </label>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// Live monitoring: a microphone (for example one placed in front of an amp) through the same effects.
export function LiveMic({ fx, onAvailable }) {
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');
  const [running, setRunning] = useState(false);
  const [gain, setGain] = useState(1);
  const [level, setLevel] = useState(0);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState('');
  const live = useRef(null);
  const fxRef = useRef(fx);
  fxRef.current = fx;

  const refreshDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === 'audioinput'));
    } catch {
      // The list simply stays as it was.
    }
  };

  const stop = () => {
    const current = live.current;
    live.current = null;
    if (current) {
      cancelAnimationFrame(current.frame);
      current.stream.getTracks().forEach((t) => t.stop());
      current.context.close().catch(() => {});
    }
    setRunning(false);
    setLevel(0);
  };

  const start = async (id = deviceId) => {
    stop();
    setError('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser cannot capture a microphone here.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: id ? { exact: id } : undefined,
          echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1,
        },
      });
      const context = new AudioContext({ latencyHint: 'interactive' });
      const input = context.createGain();
      input.gain.value = gain;
      const meter = context.createAnalyser();
      meter.fftSize = 1024;
      const rack = createRack(context);
      for (const [fxId, state] of Object.entries(fxRef.current)) rack.set(fxId, state);
      context.createMediaStreamSource(stream).connect(input);
      input.connect(meter);
      input.connect(rack.input);
      rack.output.connect(context.destination);
      rack.ready.then(() => onAvailable?.(rack.available()));
      const samples = new Float32Array(meter.fftSize);
      const entry = { stream, context, input, rack, frame: 0 };
      const tick = () => {
        meter.getFloatTimeDomainData(samples);
        let peak = 0;
        for (const v of samples) peak = Math.max(peak, Math.abs(v));
        setLevel(peak);
        entry.frame = requestAnimationFrame(tick);
      };
      entry.frame = requestAnimationFrame(tick);
      live.current = entry;
      await context.resume();
      setLatency(Math.round(1000 * ((context.baseLatency || 0) + (context.outputLatency || 0))));
      setRunning(true);
      await refreshDevices();
      const active = stream.getAudioTracks()[0]?.getSettings().deviceId;
      if (active) setDeviceId(active);
    } catch (err) {
      stop();
      setError(err.name === 'NotAllowedError'
        ? 'Microphone access was blocked. Allow it for this site and try again.'
        : err.name === 'NotFoundError' ? 'No microphone was found.' : `Could not start the microphone: ${err.message}`);
    }
  };

  useEffect(() => {
    refreshDevices();
    return stop;
  }, []);

  useEffect(() => {
    const current = live.current;
    if (!current) return;
    for (const [fxId, state] of Object.entries(fx)) current.rack.set(fxId, state);
  }, [fx]);

  const changeGain = (value) => {
    setGain(value);
    if (live.current) live.current.input.gain.value = value;
  };

  return (
    <div className="mic">
      <div className="row">
        <select aria-label="Microphone" value={deviceId}
          onChange={(e) => { setDeviceId(e.target.value); if (running) start(e.target.value); }}>
          <option value="">Default microphone</option>
          {devices.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Microphone ${i + 1}`}</option>)}
        </select>
        {running
          ? <button type="button" onClick={stop}>Stop live mic</button>
          : <button type="button" onClick={() => start()}>Start live mic</button>}
        <label className="fx-param mic-gain">
          <span>Input gain</span>
          <input type="range" min="0" max="4" step="0.05" value={gain} onChange={(e) => changeGain(Number(e.target.value))} />
        </label>
      </div>
      <div className="meter" role="img" aria-label="Microphone level">
        <div className="meter-bar" style={{ width: `${Math.min(100, Math.round(level * 140))}%`, background: level > 0.9 ? '#e05050' : '#5b86b5' }} />
      </div>
      <p className="status">
        {running
          ? `Live${latency != null ? ` · about ${latency} ms of delay` : ''}. The mic is going through the effects above.`
          : 'Point a mic at your amp and press Start.'}
        {' '}Wear headphones: speakers would feed back into the mic.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
