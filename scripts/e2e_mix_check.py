"""End-to-end check on produced mixes (plan tier "Mixed song").

For each chosen capture: play held-out (black guitar) notes through it, add bass, drums and room reverb, upload the
mix to the real server, isolate the guitar with real Demucs, match, and report where the true capture ranks.

  python scripts/e2e_mix_check.py [--models=DIR] [--bank=DIR;DIR] [--truth=name1,name2] [--samples=DIR]

The server is started here (a short-lived child process) with NAM_MODELS_DIR / NAM_DI_BANK set.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench_lib as B  # noqa: E402
from nam_tone_match.effects import load_effect, render_effect, resample  # noqa: E402
from nam_tone_match.encode import write_wav  # noqa: E402

PORT = 3290


def opt(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f'--{name}='):
            return a.split('=', 1)[1]
    return default


def make_mix(guitar_48k, seed):
    rng = np.random.default_rng(seed)
    rate = 48000
    n = len(guitar_48k)
    t = np.arange(n) / rate
    bass = 0.22 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
    drums = np.zeros(n)
    for k in range(0, n, rate // 2):
        m = min(5000, n - k)
        drums[k:k + m] += 0.45 * np.sin(2 * np.pi * 60 * np.arange(m) / rate) * np.exp(-np.arange(m) / 900)
        h = min(2000, n - k)
        drums[k:k + h] += 0.07 * rng.standard_normal(h) * np.exp(-np.arange(h) / 350)
    # a small room: exponentially decaying noise impulse response, 25% wet on the guitar
    ir = rng.standard_normal(int(0.35 * rate)) * np.exp(-np.arange(int(0.35 * rate)) / (0.08 * rate))
    ir /= np.abs(ir).sum() * 0.5
    wet = np.convolve(guitar_48k, ir)[:n]
    guitar = guitar_48k + 0.25 * wet
    mix = guitar + bass + drums
    return resample(mix / np.abs(mix).max() * 0.8, rate, 44100)


def call(method, url, data=None, headers=None, timeout=1800):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    models = os.path.abspath(opt('models', os.path.join(ROOT, 'data', 'm17')))
    samples = opt('samples', B.DEFAULT_SAMPLES)
    bank = opt('bank')
    names = sorted(n for n in os.listdir(models) if n.endswith('.nam'))
    truth = opt('truth')
    truth = truth.split(',') if truth else ['LT_MESA', '5152 Lead', 'L2 G6', 'TS9', 'DirtyShirley', 'Jet City G4', 'BUGERA']
    chosen = [next(n for n in names if t in n) for t in truth]
    env = dict(os.environ, PORT=str(PORT), NAM_MODELS_DIR=models)
    if bank:
        env['NAM_DI_BANK'] = bank
    server = subprocess.Popen(['node', os.path.join(ROOT, 'apps', 'api', 'src', 'server.js')], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{PORT}'
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(base + '/', timeout=1)
                break
            except Exception as e:
                if getattr(e, 'code', None):          # server answered (404 for missing web build is fine)
                    break
                time.sleep(0.3)
        ranks = []
        for i, name in enumerate(chosen):
            cap = (name, os.path.join(models, name), '')
            guitar = render_effect(B.recorded_phrase(samples, 'black', 5001 + i), 48000, load_effect(cap[1]), 6.0)
            guitar = guitar / np.abs(guitar).max() * 0.5
            mix = make_mix(guitar, i)
            wav = os.path.join(ROOT, 'data', f'e2e-mix-{i}.wav')
            write_wav(wav, mix, 44100, channels=2)
            t0 = time.time()
            up = call('POST', base + '/api/tracks', open(wav, 'rb').read(), {'x-filename': 'mix.wav'})
            iso = call('POST', f"{base}/api/tracks/{up['id']}/isolate",
                       json.dumps({'startSec': 1.0, 'endSec': up['durationSec'] - 1.0}).encode())
            res = call('POST', f"{base}/api/tracks/{up['id']}/match", json.dumps({'revision': iso['revision']}).encode())
            order = [r['name'] for r in res['results']]
            rank = order.index(name) + 1
            ranks.append(rank)
            top = res['results'][0]
            print(f"truth rank {rank:2d}/{len(order)}  d={res['results'][rank - 1]['distance']:.2f} dB  "
                  f"top={top['name'][:34]} d={top['distance']:.2f}  skipped={len(res['skipped'])}  "
                  f"({time.time() - t0:.0f}s)  {name[:40]}", flush=True)
        r = np.array(ranks)
        print(f'\nmixed songs: top-1 {np.mean(r == 1) * 100:.0f}%  top-3 {np.mean(r <= 3) * 100:.0f}%  '
              f'mean rank {r.mean():.1f}/{len(names)}  ({len(r)} songs; anecdotal, not a benchmark)')
    finally:
        server.kill()


if __name__ == '__main__':
    main()
