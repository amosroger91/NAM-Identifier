import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match import cli, isolate
from nam_tone_match.encode import write_wav


class TestIsolate(unittest.TestCase):
    def test_demucs_command(self):
        cmd = isolate.demucs_command('python', 'in.wav', 'out')
        self.assertIn('htdemucs_6s', cmd)
        self.assertIn('guitar', cmd)
        self.assertEqual(cmd[-1], 'in.wav')

    def test_stem_picker_from_fake_tree(self):
        with tempfile.TemporaryDirectory() as root:
            track = os.path.join(root, 'htdemucs_6s', 'region')
            os.makedirs(track)
            for name in ('no_guitar.wav', 'guitar.wav', 'vocals.wav'):
                open(os.path.join(track, name), 'w').close()
            found = isolate.find_guitar_stem(root)
            self.assertEqual(os.path.basename(found), 'guitar.wav')
            self.assertIsNone(isolate.find_guitar_stem(os.path.join(root, 'empty')))

    def test_cli_reports_demucs_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'a.source')
            write_wav(src, np.zeros(8000), 8000)
            request = {'command': 'isolate', 'path': src, 'startSec': 0.1, 'endSec': 0.5,
                       'out': os.path.join(tmp, 'g.wav')}
            out = io.StringIO()
            with mock.patch.object(isolate, 'import_demucs', side_effect=ImportError('no demucs')):
                code = cli.main(io.StringIO(json.dumps(request)), out)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue()), {'error': 'demucs_not_installed'})

    def test_cli_probe_peaks_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'a.source')
            write_wav(src, 0.5 * np.ones(16000), 8000)
            out = io.StringIO()
            self.assertEqual(cli.main(io.StringIO(json.dumps({'command': 'probe', 'path': src})), out), 0)
            self.assertEqual(json.loads(out.getvalue())['durationSec'], 2.0)
            out = io.StringIO()
            cli.main(io.StringIO(json.dumps({'command': 'peaks', 'path': src})), out)
            self.assertEqual(len(json.loads(out.getvalue())['peaks']), 1000)
            out = io.StringIO()
            self.assertEqual(cli.main(io.StringIO('{"command": "nope"}'), out), 1)
            self.assertIn('error', json.loads(out.getvalue()))
            out = io.StringIO()
            self.assertEqual(cli.main(io.StringIO('not json'), out), 1)


if __name__ == '__main__':
    unittest.main()
