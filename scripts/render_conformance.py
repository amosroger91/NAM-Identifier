"""Renderer conformance: does the converted legacy (v0.5) config render the same audio as an older `nam`
release that reads v0.5 files natively?

Usage:  python scripts/render_conformance.py [models_dir] [--old DIR]
Install the old release once with:
    python -m pip install --no-deps --target data/oldnam neural-amp-modeler==0.10.0
Selects legacy 0.5.x WaveNet files only. Requires the explicit reference installation;
it never falls back to a globally installed NAM package.
"""
import argparse, ast, json, os, subprocess, sys, tempfile
from importlib.metadata import distributions
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r'''
import sys, types, json, os, numpy as np
old, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, old)
from pathlib import Path
package = (Path(old).resolve() / 'nam')
if not (package / '__init__.py').is_file():
    raise RuntimeError('explicit reference NAM package is missing')
for key in [k for k in sys.modules if k == 'nam' or k.startswith('nam.')]:
    del sys.modules[key]
pkg = types.ModuleType('nam'); pkg.__path__ = [str(package)]; sys.modules['nam'] = pkg
from nam._version import __version__
if __version__ != '0.10.0':
    raise RuntimeError('reference NAM must be version 0.10.0')
for stub in ('nam.data', 'nam.models.losses', 'auraloss', 'auraloss.freq', 'auraloss.time', 'auraloss.perceptual'):
    sys.modules[stub] = types.ModuleType(stub)
sys.modules['nam.data'].wav_to_tensor = None
import torch
from nam.models import _from_nam
if not Path(_from_nam.__file__).resolve().is_relative_to(package):
    raise RuntimeError('reference NAM resolved outside the explicit package')
init_from_nam = _from_nam.init_from_nam
cfg = json.load(open(path))
m = init_from_nam(cfg); m.eval()
x = torch.tensor(np.load(sys.argv[4]), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    y = m(x, pad_start=True)
np.save(out, y.numpy().reshape(-1))
'''


def validate_reference(directory):
    directory = Path(directory).resolve()
    package = directory / 'nam'
    for name in ('__init__.py', '_version.py', 'models/_from_nam.py'):
        path = package / name
        if not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError(f'explicit reference NAM package is missing or outside --old: {path}')
    version = None
    for node in ast.parse((package / '_version.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            version = ast.literal_eval(node.value)
    versions = [d.version for d in distributions(path=[str(directory)])
                if d.metadata.get('Name', '').lower().replace('_', '-') == 'neural-amp-modeler']
    if version != '0.10.0' or versions != ['0.10.0']:
        raise ValueError(f'--old must contain neural-amp-modeler 0.10.0; package={version!r}, metadata={versions}')
    return str(directory)


def select_models(directory):
    selected = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() != '.nam':
            continue
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(raw, dict):
            raise ValueError(f'{path.name}: expected a NAM JSON object')
        version = str(raw.get('version', ''))
        if raw.get('architecture') == 'WaveNet' and (version == '0.5' or version.startswith('0.5.')):
            selected.append(path)
    return selected


def compare_outputs(ref, mine):
    ref, mine = np.asarray(ref, dtype=np.float64), np.asarray(mine, dtype=np.float64)
    if ref.size == 0 or mine.size == 0:
        raise ValueError('empty renderer output')
    if ref.ndim != 1 or mine.ndim != 1 or not np.isfinite(ref).all() or not np.isfinite(mine).all():
        raise ValueError('renderer output must be finite mono samples')
    k = min(len(ref), len(mine))
    with np.errstate(over='raise', invalid='raise'):
        err = float(np.max(np.abs(ref[:k] - mine[:k])))
        esr = float(np.sum((ref[:k] - mine[:k]) ** 2) / (np.sum(ref[:k] ** 2) + 1e-20))
    return err < 1e-4 and len(ref) == len(mine), err, esr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models_dir', nargs='?', default=os.path.join(ROOT, 'models'))
    parser.add_argument('--old', default=os.path.join(ROOT, 'data', 'oldnam'), help='explicit NAM 0.10.0 target installation')
    args = parser.parse_args(argv)
    try:
        old = validate_reference(args.old)
        names = select_models(args.models_dir)
    except (OSError, ValueError, SyntaxError) as exc:
        print(f'CONFIGURATION-FAIL: {exc}\ncompared 0 capture(s)')
        return 2
    if not names:
        print('SELECTION-FAIL: no legacy 0.5 WaveNet captures selected\ncompared 0 capture(s)')
        return 2
    print(f'Reference NAM 0.10.0: {old}; selected {len(names)} legacy capture(s)')
    sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))
    from nam_tone_match.effects import load_effect, render_effect
    from nam_tone_match.match import make_probe_di
    x = make_probe_di(48000, 4.0).astype(np.float32) * 0.5
    worst = 0.0
    bad = 0
    compared = 0
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, 'x.npy'), x)
        for path in names:
            n = path.name
            out = os.path.join(tmp, 'ref.npy')
            r = subprocess.run([sys.executable, '-c', CHILD, old, str(path), out, os.path.join(tmp, 'x.npy')],
                               capture_output=True, text=True)
            if r.returncode != 0:
                lines = (r.stderr or r.stdout or '').strip().splitlines()
                detail = lines[-1][:100] if lines else f'exit {r.returncode} without diagnostics'
                print(f'REFERENCE-FAIL {n[:50]}: {detail}')
                bad += 1
                continue
            try:
                ref = np.load(out, allow_pickle=False)
                mine = render_effect(x.astype(np.float64), 48000, load_effect(str(path)))
                ok, err, esr = compare_outputs(ref, mine)
            except Exception as exc:
                print(f'RENDER-FAIL {n[:50]}: {type(exc).__name__}: {exc}')
                bad += 1
                continue
            compared += 1
            bad += not ok
            worst = max(worst, err)
            print(f"{'OK  ' if ok else 'DIFF'} maxerr {err:.2e} esr {esr:.2e} len {len(ref)}/{len(mine)} {n[:55]}")
    print(f'\ncompared {compared} capture(s); worst abs error {worst:.2e}; {bad} failing capture(s)')
    return 1 if bad or compared == 0 else 0


if __name__ == '__main__':
    sys.exit(main())
