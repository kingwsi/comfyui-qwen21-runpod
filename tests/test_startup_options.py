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
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif sys.argv[1:]==["-"]:sys.stdin.read()\nif "convert_outfit_lora.py" in sys.argv[1] and os.environ.get("FAIL_CONVERT"):sys.exit(7)\n')
            fake.chmod(0o755)
            # Portable equivalent of the only flock invocation used here.
            # Keep the real advisory lock on inherited fd 9 on macOS as well.
            flock = root / 'flock'
            flock.write_text('#!/usr/bin/env python3\nimport fcntl,sys\nfcntl.flock(int(sys.argv[-1]),fcntl.LOCK_EX|fcntl.LOCK_NB)\n')
            flock.chmod(0o755)
            env = {**os.environ, 'COMFY_DIR':str(comfy), 'MODEL_DIR':str(root/'models'),
                   'RUNTIME_DIR':str(root/'runtime'), 'PYTHON_BIN':str(fake), 'CALL_LOG':str(root/'calls'),
                   'PATH':str(root)+os.pathsep+os.environ['PATH']}
            env.pop('DOWNLOAD_MODELS', None); env.pop('DOWNLOAD_LORA', None); env.pop('MODEL_VARIANT', None)
            env.update(options)
            result = subprocess.run(['bash', str(ROOT/'start.sh')], env=env, capture_output=True, text=True)
            calls = [json.loads(x) for x in (root/'calls').read_text().splitlines()] if (root/'calls').exists() else []
            return result, calls

    def test_default_does_not_touch_optional_lora(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),3)
        self.assertIn('model-manifest.json', calls[1][2])
        self.assertEqual(calls[-1][0], 'main.py')
        self.assertNotIn('lora-manifest.json',str(calls))

    def test_enabled_converts_separate_file(self):
        result, calls = self.launch(DOWNLOAD_LORA='1')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),5)
        self.assertIn('lora-manifest.json', calls[2][2])
        self.assertTrue(calls[3][0].endswith('/convert_outfit_lora.py'))
        self.assertTrue(calls[3][-1].endswith('/OutfitSwap-LoRA-GGUF-compatible.safetensors'))
        self.assertIn('--base-header',calls[3])
        self.assertEqual(calls[-1][0], 'main.py')

    def test_offline_base_needs_no_lora(self):
        result, calls = self.launch(DOWNLOAD_MODELS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verify-only',calls[1])
        self.assertEqual(len(calls),3)

    def test_offline_enabled_verifies_both(self):
        result, calls = self.launch(DOWNLOAD_MODELS='0', DOWNLOAD_LORA='1')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verify-only',calls[1]); self.assertIn('--verify-only',calls[2])

    def test_conversion_failure_stops_enabled_path(self):
        result, calls = self.launch(DOWNLOAD_LORA='1', FAIL_CONVERT='1')
        self.assertEqual(result.returncode,7)
        self.assertNotEqual(calls[-1][0], 'main.py')

    def test_invalid_option_fails_before_download(self):
        result, calls = self.launch(DOWNLOAD_LORA='true')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(calls,[])

    def test_turbo_downloads_only_turbo_manifest(self):
        result,calls=self.launch(MODEL_VARIANT='turbo')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),3)
        self.assertTrue(calls[1][2].endswith('/turbo-model-manifest.json'))
        self.assertNotIn('convert_outfit_lora.py',str(calls))

    def test_turbo_offline_verifies_selected_manifest(self):
        result,calls=self.launch(MODEL_VARIANT='turbo',DOWNLOAD_MODELS='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verify-only',calls[1])
        self.assertTrue(calls[1][2].endswith('/turbo-model-manifest.json'))

    def test_turbo_lora_and_invalid_variant_fail_before_download(self):
        for options in [dict(MODEL_VARIANT='turbo',DOWNLOAD_LORA='1'),dict(MODEL_VARIANT='unknown')]:
            result,calls=self.launch(**options)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(calls,[])

if __name__ == '__main__': unittest.main()
