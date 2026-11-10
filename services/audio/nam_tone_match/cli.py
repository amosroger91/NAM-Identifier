# stdin/stdout JSON CLI. One request object in, one response object out.

import json
import sys

from .decode import decode_file
from .effects import EffectError
from .encode import write_wav
from .isolate import DemucsNotInstalled, IsolateError, isolate_region
from .match import render_capture
from .peaks import compute_peaks


def _need(request, *keys):
    for key in keys:
        if request.get(key) in (None, ''):
            raise ValueError(f'missing "{key}"')


def _describe(path, include_query=False):
    from .tone import describe_samples, to_public, tone_snapshot

    samples, rate = decode_file(path)
    report = describe_samples(samples, rate)
    return tone_snapshot(report) if include_query else to_public(report)


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
        response = {'path': result['path'], 'durationSec': result['durationSec'], 'sampleRate': result['sampleRate']}
        if request.get('describeTone'):
            try:
                response['toneQuery'] = _describe(result['path'], include_query=True)
            except Exception:
                # Separation already succeeded. The caller can request a description again.
                response['toneQuery'] = None
        return response
    if command == 'match':
        _need(request, 'guitar', 'modelsDir', 'dataDir')
        from .drive import rank_captures
        guitar, rate = decode_file(request['guitar'])
        report = {}
        results = rank_captures(guitar, rate, request['modelsDir'], request['dataDir'], report,
                                request.get('diBankDirs'))
        return {'results': results, **report}
    if command == 'index':
        _need(request, 'modelsDir', 'dataDir')
        from .bank import bank_dirs_from_env, index_all
        from .drive import calibration
        from .effects import list_captures
        names = request.get('names') or list_captures(request['modelsDir'])
        dirs = request.get('diBankDirs')
        return index_all(names, request['modelsDir'], request['dataDir'],
                         bank_dirs_from_env() if dirs is None else dirs, drive_model=calibration())
    if command == 'render':
        _need(request, 'name', 'modelsDir', 'dataDir', 'out')
        samples, rate = render_capture(request['name'], request['modelsDir'], request['dataDir'],
                                       float(request.get('gainDb') or 0.0))
        write_wav(request['out'], samples, rate)
        return {'path': request['out'], 'sampleRate': rate}
    if command == 'describe':
        _need(request, 'path')
        return _describe(request['path'], include_query=request.get('includeQuery'))
    if command == 'compare':
        _need(request, 'a', 'b')
        from .tone import compare_samples
        left, left_rate = decode_file(request['a'])
        right, right_rate = decode_file(request['b'])
        return compare_samples(left, left_rate, right, right_rate)
    if command == 'tone-search':
        _need(request, 'guitar', 'modelsDir', 'dataDir')
        from .tone import TONE_QUERY_VERSION
        from .tone_search import search_from_description
        query = request.get('toneQuery')
        if not isinstance(query, dict) or query.get('version') != TONE_QUERY_VERSION:
            query = _describe(request['guitar'], include_query=True)
        result = search_from_description(query, request['modelsDir'], request['dataDir'])
        if request.get('includeQuery'):
            result['toneQuery'] = query
        return result
    if command == 'tone-index':
        _need(request, 'modelsDir', 'dataDir')
        from .effects import list_captures
        from .tone_search import index_captures
        names = request.get('names') or list_captures(request['modelsDir'])
        return index_captures(names, request['modelsDir'], request['dataDir'])
    if command == 'audition-source':
        _need(request, 'dataDir')
        from .audition import prepare_source
        return {'path': prepare_source(request['dataDir'])}
    if command == 'audition-warm':
        from .audition import warm
        warm()
        return {}
    if command == 'audition':
        _need(request, 'capture', 'source', 'out')
        from .audition import render_audition
        compare = request.get('compare') is True
        samples, rate = render_audition(request['capture'], request['source'], compare=compare)
        write_wav(request['out'], samples, rate, channels=2 if compare else 1)
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


def audition_worker():
    """Keep NAM imports warm across audition requests. Exit when the server closes stdin."""
    from contextlib import redirect_stdout

    for line in sys.stdin:
        ticket = None
        try:
            request = json.loads(line)
            ticket = request.pop('ticket')
            if request.get('command') not in ('audition-warm', 'audition-source', 'audition'):
                raise ValueError('Invalid audition request')
            with redirect_stdout(sys.stderr):
                result = handle(request)
        except Exception as exc:
            result = {'error': str(exc)}
        print(json.dumps({'ticket': ticket, 'result': result}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
