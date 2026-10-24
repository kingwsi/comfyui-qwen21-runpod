"""Launcher option tests use a fake Python; no GPU or downloads."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class StartupOptions(unittest.TestCase):
    def launch(self, **options):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comfy = root / 'comfy'; comfy.mkdir(); (comfy / 'main.py').touch()
            fake = root / 'fake-python'
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif sys.argv[1:]==["-"]:sys.stdin.read()\n')
            fake.chmod(0o755)
            # Portable equivalent of the only flock invocation used here.
            # Keep the real advisory lock on inherited fd 9 on macOS as well.
            flock = root / 'flock'
            flock.write_text('#!/usr/bin/env python3\nimport fcntl,sys\nfcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)\n')
            flock.chmod(0o755)
            env = {**os.environ, 'COMFY_DIR':str(comfy), 'MODEL_DIR':str(root/'models'),
                   'RUNTIME_DIR':str(root/'runtime'), 'PYTHON_BIN':str(fake), 'CALL_LOG':str(root/'calls'),
                   'PATH':str(root)+os.pathsep+os.environ['PATH']}
            env.pop('DOWNLOAD_MODELS', None); env.pop('MODEL_VARIANT', None)
            env.update(options)
            result = subprocess.run(['bash', str(ROOT/'start.sh')], env=env, capture_output=True, text=True)
            calls = [json.loads(x) for x in (root/'calls').read_text().splitlines()] if (root/'calls').exists() else []
            return result, calls

    def test_default_downloads_base_and_starts(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),3)
        self.assertTrue(calls[1][2].endswith('/model-manifest.json'))
        self.assertEqual(calls[-1][0], 'main.py')

    def test_offline_base_verifies_selected_manifest(self):
        result, calls = self.launch(DOWNLOAD_MODELS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verify-only',calls[1])
        self.assertEqual(len(calls),3)

    def test_invalid_options_fail_before_download(self):
        for options in [dict(DOWNLOAD_MODELS='true'),dict(MODEL_VARIANT='unknown')]:
            result, calls = self.launch(**options)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(calls,[])

    def test_turbo_downloads_only_turbo_manifest(self):
        result,calls=self.launch(MODEL_VARIANT='turbo')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),3)
        self.assertTrue(calls[1][2].endswith('/turbo-model-manifest.json'))
        self.assertEqual(calls[-1][0], 'main.py')

    def test_turbo_offline_verifies_selected_manifest(self):
        result,calls=self.launch(MODEL_VARIANT='turbo',DOWNLOAD_MODELS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verify-only',calls[1])
        self.assertTrue(calls[1][2].endswith('/turbo-model-manifest.json'))

if __name__ == '__main__': unittest.main()
