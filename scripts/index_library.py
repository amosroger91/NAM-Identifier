"""Pre-build the matching index for captures (resumable; safe to interrupt and to run several in parallel).

  python scripts/index_library.py [--models=DIR] [--di=DIR[;DIR...]] [--workers=N] [--limit=N]
  python scripts/index_library.py --tone [--models=DIR] [--workers=N] [--limit=N]

--di defaults to the NAM_DI_BANK environment variable. Indexing renders 16 phrases x 6 input trims per capture
(roughly a minute each), so index once and matching afterwards is fast.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))


def main():
    tone = '--tone' in sys.argv
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    models = os.path.abspath(opts.get('models', os.path.join(ROOT, 'models')))
    data = os.path.join(ROOT, 'data')
    workers = int(opts.get('workers', 1))
    from nam_tone_match.effects import list_captures
    names = list_captures(models)
    if not names:
        print(f'no .nam files found in {models}')
        return 1
    if opts.get('limit'):
        names = names[: int(opts['limit'])]
    di = opts.get('di')
    env = dict(os.environ)
    if di:
        env['NAM_DI_BANK'] = di
    start = time.time()
    chunks = [names[i::workers] for i in range(workers)]
    procs = []
    for chunk in chunks:
        if not chunk:
            continue
        command = 'tone-index' if tone else 'index'
        request = json.dumps({'command': command, 'modelsDir': models, 'dataDir': data, 'names': chunk})
        procs.append(subprocess.Popen([sys.executable, '-m', 'nam_tone_match'], cwd=os.path.join(ROOT, 'services', 'audio'),
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env))
        procs[-1].stdin.write(request)
        procs[-1].stdin.close()
    total = {'indexed': 0, 'cached': 0, 'failed': [], 'driveUnmeasured': 0}
    for proc in procs:
        out = proc.stdout.read()
        proc.wait()
        try:
            result = json.loads(out.strip().splitlines()[-1])
        except Exception:
            print('worker failed:', out[-300:])
            return 1
        if result.get('error'):
            print('worker failed:', result['error'])
            return 1
        total['indexed'] += result.get('indexed', 0)
        total['cached'] += result.get('cached', 0)
        total['failed'] += result.get('failed', [])
        total['driveUnmeasured'] += result.get('driveUnmeasured', 0)
        bank = result.get('bank')
    if tone:
        print(f"tone cards built {total['indexed']}, already cached {total['cached']}, "
              f"failed {len(total['failed'])} in {time.time() - start:.0f}s")
    else:
        print(f"indexed {total['indexed']}, already cached {total['cached']}, failed {len(total['failed'])} "
              f"in {time.time() - start:.0f}s; drive unavailable for {total['driveUnmeasured']}; bank: {bank}")
    for item in total['failed'][:20]:
        print('  FAILED', item['name'], '-', item['error'][:100])
    return 0


if __name__ == '__main__':
    sys.exit(main())
