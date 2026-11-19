"""Exercise the actual clone commands and retain the existing inference route."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
from humanscale.cli import main


class InstantVerificationTests(unittest.TestCase):
    def command(self, output, model=None, command='run'):
        argv = [sys.executable, '-m', 'humanscale', command, '--output', str(output)]
        if command == 'run':
            argv += ['--scenario', 'all']
        else:
            argv += ['--workspace', str(output / 'guard_workspace'), '--port', '19193']
        if model is not None:
            argv += ['--model', str(model)]
        env = {**os.environ, 'PYTHONPATH': str(RUNNER), 'PATH': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((output / 'receipt.json').read_text())
        self.assertEqual(receipt['errors'], [], receipt)
        self.assertTrue(receipt['passed'])
        self.assertTrue(receipt['runtime_healthy'])
        self.assertEqual(receipt['state'], 'done')
        self.assertEqual(receipt['reason'], 'complete')
        self.assertEqual(len(receipt['cases']), 4)
        self.assertTrue(receipt['sqlite']['fake_green_caught'])
        self.assertEqual((receipt['sqlite']['before'], receipt['sqlite']['after']), (13, 42))
        self.assertFalse(receipt['inference_performed'])
        self.assertIn('Captured Apple M5 receipts: 4/4', result.stdout)
        if sys.platform == 'darwin':
            self.assertEqual(receipt['boundary']['state'], 'HELD')
            self.assertEqual(len(receipt['boundary']['probes']), 5)
        return receipt

    def test_omitted_missing_directory_and_repeat_with_empty_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in (None, root / 'absent.gguf', root):
                self.command(root / 'receipt', model)

    def test_guard_omitted_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in (None, root / 'absent.gguf'):
                self.command(root / 'receipt', model, 'guard')

    def test_existing_model_keeps_all_four_inference_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.gguf'
            model.write_bytes(b'GGUF')
            with patch.object(sys, 'argv', ['humanscale', 'run', '--model', str(model), '--output', str(root)]), \
                    patch('humanscale.cli.run', return_value={'recomputed': {}}) as inference, \
                    patch('humanscale.cli.instant_verification') as instant, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual([call.args[0] for call in inference.call_args_list], [1, 2, 3, 4])
            instant.assert_not_called()

    def test_existing_inference_receipt_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'receipt.json'
            old.write_text('{"schema":"humanscale.receipt.v1","preserve":true}')
            result = subprocess.run([sys.executable, '-m', 'humanscale', 'run', '--output', str(root)],
                                    env={**os.environ, 'PYTHONPATH': str(RUNNER)},
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(old.read_text())['preserve'])
            self.assertTrue(json.loads((root / 'verification.receipt.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
