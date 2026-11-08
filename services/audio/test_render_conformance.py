"""Conformance diagnostic guards, without real NAM renders or downloads."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services' / 'audio'))
SPEC = importlib.util.spec_from_file_location('render_conformance_tested', ROOT / 'scripts' / 'render_conformance.py')
conformance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conformance)


class TestRenderConformance(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'services' / 'audio' / '__pycache__'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / 'old'
        package = self.old / 'nam'
        (package / 'models').mkdir(parents=True)
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / '_version.py').write_text('__version__ = "0.10.0"', encoding='utf-8')
        (package / 'models' / '_from_nam.py').write_text('', encoding='utf-8')
        self.metadata = self.old / 'neural_amp_modeler-0.10.0.dist-info' / 'METADATA'
        self.metadata.parent.mkdir()
        self.metadata.write_text('Name: neural-amp-modeler\nVersion: 0.10.0\n', encoding='utf-8')
        self.models = self.root / 'models'
        self.models.mkdir()

    def put(self, name, version='0.5.4', architecture='WaveNet'):
        path = self.models / name
        path.write_text(json.dumps({'version': version, 'architecture': architecture}), encoding='utf-8')
        return path

    def invoke(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(tempfile, 'tempdir', str(self.root)):
            code = conformance.main([str(self.models), '--old=' + str(self.old)])
        return code, output.getvalue()

    def test_explicit_reference_validates_local_package_and_metadata(self):
        self.assertEqual(conformance.validate_reference(self.old), str(self.old.resolve()))

    def test_missing_reference_cannot_fall_back_to_global_nam(self):
        with mock.patch('importlib.util.find_spec') as global_lookup:
            with self.assertRaisesRegex(ValueError, 'reference NAM package is missing'):
                conformance.validate_reference(self.root / 'missing')
        global_lookup.assert_not_called()

    def test_wrong_package_or_distribution_version_is_rejected(self):
        self.metadata.write_text('Name: neural-amp-modeler\nVersion: 0.12.0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'must contain neural-amp-modeler 0.10.0'):
            conformance.validate_reference(self.old)
        self.metadata.write_text('Name: neural-amp-modeler\nVersion: 0.10.0\n', encoding='utf-8')
        (self.old / 'nam' / '_version.py').write_text('__version__ = "0.12.0"', encoding='utf-8')
        with self.assertRaises(ValueError):
            conformance.validate_reference(self.old)

    def test_filter_selects_only_legacy_05_wavenet(self):
        expected = self.put('legacy.nam')
        self.put('modern.nam', '0.7.0')
        self.put('recurrent.nam', '0.5.4', 'LSTM')
        self.put('container.nam', '0.5.4', 'SlimmableContainer')
        self.put('other-minor.nam', '0.50.0')
        self.assertEqual(conformance.select_models(self.models), [expected])

    def test_empty_selection_fails_before_subprocess(self):
        self.put('modern.nam', '0.7.0')
        with mock.patch.object(conformance.subprocess, 'run') as run:
            code, output = self.invoke()
        self.assertNotEqual(code, 0)
        self.assertIn('compared 0', output)
        run.assert_not_called()

    def test_empty_nonfinite_or_length_mismatched_output_cannot_pass(self):
        for reference, current in (([], [0.0]), ([0.0], []), ([float('nan')], [0.0]), ([0.0], [float('inf')])):
            with self.subTest(reference=reference, current=current):
                with self.assertRaises(ValueError):
                    conformance.compare_outputs(reference, current)
        self.assertFalse(conformance.compare_outputs([0.1, 0.2], [0.1])[0])
        self.assertEqual(conformance.compare_outputs([0.1, 0.2], [0.1, 0.2]), (True, 0.0, 0.0))

    def test_no_valid_comparisons_fails_and_empty_stderr_is_safe(self):
        self.put('legacy.nam')
        failed = subprocess.CompletedProcess([], 1, stdout='', stderr='')
        with mock.patch('nam_tone_match.match.make_probe_di', return_value=np.ones(4)), \
                mock.patch.object(conformance.subprocess, 'run', return_value=failed):
            code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('without diagnostics', output)
        self.assertIn('compared 0 capture(s)', output)


if __name__ == '__main__':
    unittest.main()
