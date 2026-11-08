# Guitar isolation with Demucs (htdemucs_6s)

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

MODEL = 'htdemucs_6s'


class DemucsNotInstalled(RuntimeError):
    """The `demucs` package cannot be imported."""


class IsolateError(RuntimeError):
    """Demucs ran but did not produce a guitar stem."""


def import_demucs():
    """Import Demucs. A separate function so tests can stub it."""
    import demucs  # noqa: F401
    return demucs


def demucs_command(python_exe: str, input_wav: str, out_dir: str) -> List[str]:
    """Command that splits `input_wav` with htdemucs_6s and keeps only the guitar stem
    (output: <out_dir>/htdemucs_6s/<track>/guitar.wav)."""
    return [python_exe, '-m', 'demucs', '-n', MODEL, '--two-stems', 'guitar', '-o', out_dir, input_wav]


def find_guitar_stem(demucs_output_dir: str) -> Optional[str]:
    """Path of the guitar stem file in a Demucs output tree, or None. `no_guitar.*` never counts."""
    found = []
    for root, _dirs, files in os.walk(demucs_output_dir):
        for name in sorted(files):
            lower = name.lower()
            if 'guitar' in lower and lower.endswith(('.wav', '.mp3', '.flac')):
                found.append((lower.startswith('no_') or 'no_guitar' in lower, os.path.join(root, name)))
    found.sort(key=lambda item: item[0])
    return found[0][1] if found and not found[0][0] else None


def isolate_region(source_path: str, start_sec: float, end_sec: float, out_wav: str):
    """Cut [start_sec, end_sec] from the source, separate the guitar, write mono WAV to out_wav."""
    try:
        import_demucs()
    except ImportError:
        raise DemucsNotInstalled('demucs_not_installed')
    from .decode import decode_file, decode_wav
    from .encode import write_wav
    from .slice import slice_audio

    samples, rate = decode_file(source_path)
    piece = slice_audio(samples, start_sec, end_sec, rate)
    work = tempfile.mkdtemp(prefix='nam-isolate-')
    try:
        region = os.path.join(work, 'region.wav')
        write_wav(region, piece, rate, channels=2)
        out_dir = os.path.join(work, 'out')
        proc = subprocess.run(demucs_command(sys.executable, region, out_dir),
                              capture_output=True, text=True)
        stem = find_guitar_stem(out_dir) if proc.returncode == 0 else None
        if not stem:
            tail = (proc.stderr or proc.stdout or '').strip()[-500:]
            raise IsolateError(f'demucs failed (exit {proc.returncode}): {tail}')
        guitar, guitar_rate = decode_wav(stem)
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
        write_wav(out_wav, guitar, guitar_rate)
        return {'path': out_wav, 'sampleRate': guitar_rate, 'durationSec': len(guitar) / guitar_rate}
    finally:
        shutil.rmtree(work, ignore_errors=True)
