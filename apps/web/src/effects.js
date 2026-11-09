// Optional worship-style effects for the audition player, built from Web Audio nodes.
// Signal flow (mono in): compressor -> overdrive -> chorus -> phaser -> tremolo -> volume swell
//   -> dotted delay -> reverb sends (spring, hall, plate, shimmer) -> output.
// Reverbs are generated impulse responses and the overdrives are shaped curves: they are models of
// the usual sounds, not captures of specific hardware.

export const EFFECTS = [
  { id: 'compressor', name: 'Compressor', hint: 'Evens out picked parts', params: [
    { key: 'amount', label: 'Squeeze', min: 0, max: 1, step: 0.01, value: 0.5 }] },
  { id: 'drive', name: 'Overdrive', hint: 'Klon / Screamer / fuzz', select: {
    key: 'type', value: 'klon', options: [['klon', 'Klon-style'], ['screamer', 'Tube Screamer'], ['fuzz', 'Fuzz']] },
  params: [{ key: 'drive', label: 'Drive', min: 0, max: 1, step: 0.01, value: 0.35 }] },
  { id: 'chorus', name: 'Chorus', params: [
    { key: 'rate', label: 'Rate', min: 0.1, max: 3, step: 0.05, value: 0.8 },
    { key: 'depth', label: 'Depth', min: 0, max: 1, step: 0.01, value: 0.5 }] },
  { id: 'phaser', name: 'Phaser', params: [
    { key: 'rate', label: 'Rate', min: 0.05, max: 2, step: 0.05, value: 0.4 }] },
  { id: 'tremolo', name: 'Tremolo', params: [
    { key: 'rate', label: 'Rate', min: 0.5, max: 10, step: 0.1, value: 4.5 },
    { key: 'depth', label: 'Depth', min: 0, max: 1, step: 0.01, value: 0.5 }] },
  { id: 'swell', name: 'Volume swell', hint: 'Slow attack on every note', needsWorklet: true, params: [
    { key: 'attack', label: 'Attack (s)', min: 0.1, max: 2.5, step: 0.05, value: 0.6 }] },
  { id: 'delay', name: 'Dotted-eighth delay', hint: 'The classic rhythmic echo', params: [
    { key: 'tempo', label: 'Tempo (BPM)', min: 60, max: 160, step: 1, value: 72 },
    { key: 'repeats', label: 'Repeats', min: 0, max: 0.8, step: 0.01, value: 0.4 },
    { key: 'mix', label: 'Mix', min: 0, max: 1, step: 0.01, value: 0.35 }] },
  { id: 'spring', name: 'Spring reverb', params: [
    { key: 'mix', label: 'Mix', min: 0, max: 1, step: 0.01, value: 0.3 }] },
  { id: 'hall', name: 'Hall reverb', params: [
    { key: 'mix', label: 'Mix', min: 0, max: 1, step: 0.01, value: 0.3 }] },
  { id: 'plate', name: 'EMT-style plate reverb', params: [
    { key: 'mix', label: 'Mix', min: 0, max: 1, step: 0.01, value: 0.45 }] },
  { id: 'shimmer', name: 'Shimmer reverb', hint: 'Octave-up reverb pad', needsWorklet: true, params: [
    { key: 'mix', label: 'Mix', min: 0, max: 1, step: 0.01, value: 0.3 }] },
];

export function defaultEffectState() {
  const state = {};
  for (const fx of EFFECTS) {
    state[fx.id] = { on: false };
    if (fx.select) state[fx.id][fx.select.key] = fx.select.value;
    for (const p of fx.params) state[fx.id][p.key] = p.value;
  }
  return state;
}

// --------------------------------------------------------------------------- worklets
const WORKLET_SOURCE = `
class PitchShift extends AudioWorkletProcessor {
  static get parameterDescriptors() { return [{ name: 'ratio', defaultValue: 2, minValue: 0.25, maxValue: 4 }]; }
  constructor() {
    super();
    this.win = 4096;
    this.buf = new Float32Array(this.win * 4);
    this.w = 0;
    this.d = 0;
  }
  process(inputs, outputs, parameters) {
    const out = outputs[0][0];
    if (!out) return true;
    const input = inputs[0] && inputs[0][0];
    const ratio = parameters.ratio[0];
    const win = this.win;
    const size = this.buf.length;
    for (let i = 0; i < out.length; i += 1) {
      this.buf[this.w] = input ? input[i] : 0;
      this.d += 1 - ratio;
      if (this.d < 0) this.d += win;
      else if (this.d >= win) this.d -= win;
      let d2 = this.d + win / 2;
      if (d2 >= win) d2 -= win;
      const g1 = 0.5 - 0.5 * Math.cos(2 * Math.PI * this.d / win);
      const g2 = 0.5 - 0.5 * Math.cos(2 * Math.PI * d2 / win);
      out[i] = g1 * this.read(this.d, size) + g2 * this.read(d2, size);
      this.w = (this.w + 1) % size;
    }
    return true;
  }
  read(delay, size) {
    let pos = this.w - delay;
    if (pos < 0) pos += size;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    return this.buf[i0 % size] * (1 - frac) + this.buf[(i0 + 1) % size] * frac;
  }
}
registerProcessor('pitch-shift', PitchShift);

class Swell extends AudioWorkletProcessor {
  static get parameterDescriptors() { return [{ name: 'attack', defaultValue: 0.9, minValue: 0.05, maxValue: 3 }]; }
  constructor() {
    super();
    this.fast = 0;
    this.slow = 0;
    this.age = 99;
    this.quiet = 0;
  }
  process(inputs, outputs, parameters) {
    const input = inputs[0] && inputs[0][0];
    const out = outputs[0][0];
    if (!out) return true;
    const attack = Math.max(0.05, parameters.attack[0]) * sampleRate;
    const holdoff = 0.12 * sampleRate;
    for (let i = 0; i < out.length; i += 1) {
      const x = input ? input[i] : 0;
      const level = Math.abs(x);
      this.fast += (level - this.fast) * (level > this.fast ? 0.05 : 0.002);
      this.slow += (level - this.slow) * 0.0004;
      this.age += 1;
      // A new pluck: the quick envelope jumps well above the slow one after a quiet moment.
      if (this.fast > 0.01 && this.fast > 2.2 * this.slow && this.age > holdoff) this.age = 0;
      const ramp = Math.min(1, this.age / attack);
      out[i] = x * ramp * ramp * (3 - 2 * ramp);
    }
    return true;
  }
}
registerProcessor('swell', Swell);
`;

// --------------------------------------------------------------------------- impulse responses
function noiseImpulse(context, { seconds, preDelay = 0.004, damping, rumble = 0.02, stereoSeed = 1 }) {
  const rate = context.sampleRate;
  const length = Math.floor(rate * seconds);
  const pre = Math.floor(rate * preDelay);
  const buffer = context.createBuffer(2, length, rate);
  for (let channel = 0; channel < 2; channel += 1) {
    const data = buffer.getChannelData(channel);
    let low = 0;
    let slow = 0;
    let seed = (channel + 1) * 7919 * stereoSeed;
    for (let i = pre; i < length; i += 1) {
      const t = (i - pre) / rate;
      seed = (seed * 16807) % 2147483647;
      const white = (seed / 2147483647) * 2 - 1;
      const decay = 10 ** (-3 * t / (seconds * 0.9));
      const coefficient = Math.min(0.97, damping(t / seconds));
      low += (white - low) * (1 - coefficient);   // highs die faster than lows
      slow += (low - slow) * rumble;               // remove sub rumble
      data[i] = (low - slow) * decay;
    }
  }
  return buffer;
}

export const plateImpulse = (context) => noiseImpulse(context, {
  seconds: 2.2, preDelay: 0.004, damping: (x) => 0.25 + 0.7 * x,
});

const hallImpulse = (context, seconds = 3.6) => noiseImpulse(context, {
  seconds, preDelay: 0.022, damping: (x) => 0.55 + 0.4 * x, rumble: 0.01, stereoSeed: 3,
});

function springImpulse(context) {
  const rate = context.sampleRate;
  const seconds = 2.0;
  const length = Math.floor(rate * seconds);
  const buffer = context.createBuffer(2, length, rate);
  for (let channel = 0; channel < 2; channel += 1) {
    const data = buffer.getChannelData(channel);
    const interval = 0.034 + channel * 0.004;
    const chirp = Math.floor(rate * 0.012);
    for (let k = 0; k * interval < seconds - 0.05; k += 1) {
      const start = Math.floor(k * interval * rate);
      const gain = 0.82 ** k;
      for (let i = 0; i < chirp && start + i < length; i += 1) {
        const x = i / chirp;                        // dispersion: a short descending "drip"
        const hz = 3600 - 3000 * x;
        data[start + i] += gain * Math.sin(2 * Math.PI * hz * (i / rate)) * Math.sin(Math.PI * x);
      }
    }
    let seed = 99 + channel;
    let low = 0;
    for (let i = 0; i < length; i += 1) {
      seed = (seed * 16807) % 2147483647;
      low += ((seed / 2147483647) * 2 - 1 - low) * 0.5;
      data[i] += 0.12 * low * 10 ** (-3 * (i / rate) / 1.6);
    }
  }
  return buffer;
}

function curve(k) {
  const samples = new Float32Array(2048);
  for (let i = 0; i < samples.length; i += 1) {
    const x = (i / (samples.length - 1)) * 2 - 1;
    samples[i] = Math.tanh(k * x) / Math.tanh(k);
  }
  return samples;
}

// --------------------------------------------------------------------------- rack
const smooth = (context, param, value, time = 0.015) => {
  param.cancelScheduledValues(context.currentTime);
  param.setTargetAtTime(value, context.currentTime, time);
};

function insert(context, build) {
  const input = context.createGain();
  const output = context.createGain();
  const dry = context.createGain();
  const wet = context.createGain();
  wet.gain.value = 0;
  input.connect(dry);
  dry.connect(output);
  const fx = build(input);
  fx.out.connect(wet);
  wet.connect(output);
  return {
    input,
    output,
    fx,
    set(state) {
      smooth(context, dry.gain, state.on ? 0 : 1);
      smooth(context, wet.gain, state.on ? 1 : 0);
      fx.update(state);
    },
  };
}

function buildCompressor(context) {
  return (input) => {
    const comp = context.createDynamicsCompressor();
    comp.attack.value = 0.008;
    comp.release.value = 0.25;
    comp.knee.value = 18;
    const makeup = context.createGain();
    input.connect(comp);
    comp.connect(makeup);
    return {
      out: makeup,
      update({ amount }) {
        comp.threshold.value = -14 - 30 * amount;
        comp.ratio.value = 2 + 10 * amount;
        makeup.gain.value = 1 + 0.5 * amount;
      },
    };
  };
}

const DRIVE_MODES = {
  klon: { k: 1.4, boost: 7, eqHz: 1600, eqDb: 3, clean: 0.55, tone: 7000, level: 0.5 },
  screamer: { k: 2.2, boost: 16, eqHz: 720, eqDb: 9, clean: 0, tone: 5200, level: 0.32 },
  fuzz: { k: 7, boost: 60, eqHz: 900, eqDb: 0, clean: 0, tone: 3800, level: 0.16 },
};

function buildDrive(context) {
  return (input) => {
    const eq = context.createBiquadFilter();
    eq.type = 'peaking';
    eq.Q.value = 0.9;
    const pre = context.createGain();
    const shaper = context.createWaveShaper();
    shaper.oversample = '4x';
    const tone = context.createBiquadFilter();
    tone.type = 'lowpass';
    const level = context.createGain();
    const clean = context.createGain();
    const out = context.createGain();
    input.connect(eq);
    eq.connect(pre);
    pre.connect(shaper);
    shaper.connect(tone);
    tone.connect(level);
    level.connect(out);
    input.connect(clean);
    clean.connect(out);
    const curves = {};
    return {
      out,
      update({ type, drive }) {
        const mode = DRIVE_MODES[type] || DRIVE_MODES.klon;
        curves[type] ??= curve(mode.k);
        shaper.curve = curves[type];
        eq.frequency.value = mode.eqHz;
        eq.gain.value = mode.eqDb;
        pre.gain.value = 1 + drive * mode.boost;
        tone.frequency.value = mode.tone;
        level.gain.value = mode.level;
        clean.gain.value = mode.clean;
      },
    };
  };
}

function buildChorus(context) {
  return (input) => {
    const out = context.createGain();
    const direct = context.createGain();
    direct.gain.value = 0.7;
    input.connect(direct);
    direct.connect(out);
    const voices = [0.012, 0.019].map((base, index) => {
      const delay = context.createDelay(0.06);
      delay.delayTime.value = base;
      const lfo = context.createOscillator();
      const amount = context.createGain();
      lfo.frequency.value = 0.8 * (1 + index * 0.13);
      lfo.connect(amount);
      amount.connect(delay.delayTime);
      lfo.start();
      const voice = context.createGain();
      voice.gain.value = 0.5;
      input.connect(delay);
      delay.connect(voice);
      voice.connect(out);
      return { lfo, amount, index };
    });
    return {
      out,
      update({ rate, depth }) {
        for (const { lfo, amount, index } of voices) {
          lfo.frequency.value = rate * (1 + index * 0.13);
          amount.gain.value = 0.004 * depth;
        }
      },
    };
  };
}

function buildPhaser(context) {
  return (input) => {
    const out = context.createGain();
    const direct = context.createGain();
    direct.gain.value = 0.6;
    input.connect(direct);
    direct.connect(out);
    const lfo = context.createOscillator();
    const sweep = context.createGain();
    sweep.gain.value = 500;
    lfo.connect(sweep);
    lfo.start();
    let node = input;
    for (let i = 0; i < 4; i += 1) {
      const stage = context.createBiquadFilter();
      stage.type = 'allpass';
      stage.frequency.value = 900;
      stage.Q.value = 0.7;
      sweep.connect(stage.frequency);
      node.connect(stage);
      node = stage;
    }
    const wet = context.createGain();
    wet.gain.value = 0.6;
    node.connect(wet);
    wet.connect(out);
    return { out, update({ rate }) { lfo.frequency.value = rate; } };
  };
}

function buildTremolo(context) {
  return (input) => {
    const amp = context.createGain();
    const lfo = context.createOscillator();
    const amount = context.createGain();
    lfo.connect(amount);
    amount.connect(amp.gain);
    lfo.start();
    input.connect(amp);
    return {
      out: amp,
      update({ rate, depth }) {
        lfo.frequency.value = rate;
        amp.gain.value = 1 - depth / 2;
        amount.gain.value = depth / 2;
      },
    };
  };
}

export function createRack(context) {
  const state = defaultEffectState();
  const input = context.createGain();
  const output = context.createGain();
  // Safety limiter so stacked effects and drives never clip the speakers.
  const limiter = context.createDynamicsCompressor();
  limiter.threshold.value = -3;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.002;
  limiter.release.value = 0.1;
  output.connect(limiter);
  const inserts = {};
  const builders = {
    compressor: buildCompressor, drive: buildDrive, chorus: buildChorus,
    phaser: buildPhaser, tremolo: buildTremolo,
  };
  let node = input;
  for (const id of ['compressor', 'drive', 'chorus', 'phaser', 'tremolo', 'swell']) {
    if (id === 'swell') {
      // The swell is filled in once the worklet loads; until then it passes audio through.
      const slot = context.createGain();
      inserts.swell = insert(context, (source) => {
        source.connect(slot);
        return { out: slot, update() {} };
      });
      inserts.swell.slot = slot;
    } else {
      inserts[id] = insert(context, builders[id](context));
    }
    node.connect(inserts[id].input);
    node = inserts[id].output;
  }

  const post = context.createGain();
  node.connect(post);
  post.connect(output);

  // Dotted-eighth delay with a darkened feedback path.
  const delay = context.createDelay(2);
  const tone = context.createBiquadFilter();
  tone.type = 'lowpass';
  tone.frequency.value = 3600;
  const feedback = context.createGain();
  const delayMix = context.createGain();
  node.connect(delay);
  delay.connect(tone);
  tone.connect(feedback);
  feedback.connect(delay);
  tone.connect(delayMix);
  delayMix.connect(post);
  delayMix.gain.value = 0;

  const sends = {};
  for (const [id, make] of [['spring', springImpulse], ['hall', hallImpulse], ['plate', plateImpulse]]) {
    const convolver = context.createConvolver();
    convolver.buffer = make(context);
    const wet = context.createGain();
    wet.gain.value = 0;
    post.connect(convolver);
    convolver.connect(wet);
    wet.connect(output);
    sends[id] = wet;
  }
  const shimmerWet = context.createGain();
  shimmerWet.gain.value = 0;
  shimmerWet.connect(output);
  sends.shimmer = shimmerWet;

  const available = { swell: false, shimmer: false };

  const apply = (id) => {
    const s = state[id];
    if (inserts[id]) {
      if (id === 'swell') {
        if (!available.swell) return;
        inserts.swell.worklet.parameters.get('attack').value = s.attack;
      }
      inserts[id].set(s);
    } else if (id === 'delay') {
      smooth(context, delay.delayTime, (60 / s.tempo) * 0.75, 0.05);
      smooth(context, feedback.gain, s.repeats);
      smooth(context, delayMix.gain, s.on ? s.mix : 0);
    } else if (sends[id]) {
      if (id === 'shimmer' && !available.shimmer) return;
      smooth(context, sends[id].gain, s.on ? s.mix * (id === 'shimmer' ? 1.2 : 1) : 0);
    }
  };

  const ready = (async () => {
    if (!context.audioWorklet) return;
    const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
    try {
      await context.audioWorklet.addModule(url);
      const swell = new AudioWorkletNode(context, 'swell', { outputChannelCount: [1] });
      const { slot, input: swellIn } = inserts.swell;
      swellIn.disconnect(slot);
      swellIn.connect(swell);
      swell.connect(slot);
      inserts.swell.worklet = swell;
      available.swell = true;

      const shifter = new AudioWorkletNode(context, 'pitch-shift', { outputChannelCount: [1] });
      const reverb = context.createConvolver();
      reverb.buffer = hallImpulse(context, 4.8);
      const brighten = context.createBiquadFilter();
      brighten.type = 'lowpass';
      brighten.frequency.value = 6000;
      const loop = context.createGain();
      loop.gain.value = 0.32;
      post.connect(shifter);
      shifter.connect(reverb);
      reverb.connect(brighten);
      brighten.connect(shimmerWet);
      brighten.connect(loop);
      loop.connect(shifter);        // each pass rises another octave, then fades
      available.shimmer = true;
    } catch {
      // Without worklets the two worklet effects stay unavailable; the rest of the rack still works.
    } finally {
      URL.revokeObjectURL(url);
    }
    for (const id of Object.keys(state)) apply(id);
  })();

  for (const id of Object.keys(state)) apply(id);

  return {
    input,
    output: limiter,
    ready,
    available: () => ({ ...available }),
    set(id, patch) {
      state[id] = { ...state[id], ...patch };
      apply(id);
    },
    get state() { return state; },
  };
}
