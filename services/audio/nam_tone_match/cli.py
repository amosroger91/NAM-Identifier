# stdin/stdout JSON CLI. One request object in, one response object out.

import json
import sys

from .decode import decode_file
from .effects import EffectError
from .encode import write_wav
from .isolate import DemucsNotInstalled, IsolateError, isolate_region
from .match import rank_captures, render_capture
from .peaks import compute_peaks


def _need(request, *keys):
    for key in keys:
        if request.get(key) in (None, ''):
            raise ValueError(f'missing "{key}"')


def handle(request):
    command = request.get('command')
    if command == 'probe':
        _need(request, 'path')
        samples, rate = decode_file(request['path'])
        return {'durationSec': len(samples) / rate, 'sampleRate': rate}
    if command == 'peaks':
        _need(request, 'path')
        samples, rate = decode_file(request['path'])
        count = int(request.get('count') or 1000)
        return {'peaks': [float(p) for p in compute_peaks(samples, rate, count)],
                'durationSec': len(samples) / rate}
    if command == 'isolate':
        _need(request, 'path', 'out', 'startSec', 'endSec')
        try:
            result = isolate_region(request['path'], float(request['startSec']),
                                    float(request['endSec']), request['out'])
        except DemucsNotInstalled:
            return {'error': 'demucs_not_installed'}
        return {'path': result['path'], 'durationSec': result['durationSec'],
                'sampleRate': result['sampleRate']}
    if command == 'match':
        _need(request, 'guitar', 'modelsDir', 'dataDir')
        guitar, rate = decode_file(request['guitar'])
        return {'results': rank_captures(guitar, rate, request['modelsDir'], request['dataDir'])}
    if command == 'render':
        _need(request, 'name', 'modelsDir', 'dataDir', 'out')
        samples, rate = render_capture(request['name'], request['modelsDir'], request['dataDir'],
                                       float(request.get('gainDb') or 0.0))
        write_wav(request['out'], samples, rate)
        return {'path': request['out'], 'sampleRate': rate}
    raise ValueError(f'unknown command {command!r}')


def main(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        raw = stdin.read()
        if not raw.strip():
            raise ValueError('no input provided')
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError('request must be a JSON object')
        response = handle(request)
        code = 0
    except (ValueError, EffectError, IsolateError, OSError) as exc:
        response, code = {'error': str(exc)}, 1
    except Exception as exc:  # keep the stdout contract: always one JSON object
        response, code = {'error': f'{type(exc).__name__}: {exc}'}, 1
    stdout.write(json.dumps(response))
    stdout.write('\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
