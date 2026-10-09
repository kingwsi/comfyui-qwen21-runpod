import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_models', ROOT / 'scripts/download_models.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class Downloads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.data = b'one small test model'
        self.item = {'remote_path': 'small.gguf', 'target_path': 'unet/small.gguf', 'size_bytes': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.manifest = {'repo_id': 'test/model', 'revision': 'a'*40, 'files': [self.item]}

    def tearDown(self):
        self.temp.cleanup()

    def test_release_manifest_has_exactly_three_verified_files(self):
        data = m.load_manifest(ROOT / 'model-manifest.json')
        self.assertEqual(data['total_size_bytes'], 25801402224)

    def test_download_and_hash(self):
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(self.data)):
            m.ensure_file(self.root, self.manifest, self.item)
        self.assertEqual((self.root / self.item['target_path']).read_bytes(), self.data)

    def test_resume_respects_content_range(self):
        partial = self.root / 'small.part'
        partial.write_bytes(self.data[:4])
        response = Response(self.data[4:], 206, {'Content-Range': f'bytes 4-{len(self.data)-1}/{len(self.data)}'})
        with patch.object(m.urllib.request, 'urlopen', return_value=response) as call:
            m.stream_download('https://huggingface.co/test', partial, len(self.data))
        self.assertEqual(partial.read_bytes(), self.data)
        self.assertEqual(call.call_args.args[0].get_header('Range'), 'bytes=4-')

    def test_ignored_range_restarts_instead_of_appending(self):
        partial = self.root / 'small.part'
        partial.write_bytes(self.data[:4])
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(self.data)):
            m.stream_download('https://huggingface.co/test', partial, len(self.data))
        self.assertEqual(partial.read_bytes(), self.data)

    def test_wrong_range_is_rejected(self):
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(self.data, 206, {'Content-Range': 'bytes 2-10/20'})):
            with self.assertRaises(ValueError):
                m.stream_download('https://huggingface.co/test', self.root / 'small.part', len(self.data))

    def test_incomplete_download_is_not_promoted(self):
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(self.data[:4])):
            with self.assertRaises(OSError):
                m.stream_download('https://huggingface.co/test', self.root / 'small.part', len(self.data))
        self.assertEqual((self.root / 'small.part').stat().st_size, 4)

    def test_oversized_response_is_rejected(self):
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(self.data+b'x')):
            with self.assertRaises(ValueError):
                m.stream_download('https://huggingface.co/test', self.root / 'small.part', len(self.data))

    def test_bad_digest_is_not_promoted(self):
        with patch.object(m.urllib.request, 'urlopen', return_value=Response(b'x'*len(self.data))):
            with self.assertRaises(ValueError):
                m.ensure_file(self.root, self.manifest, self.item)
        self.assertFalse((self.root / self.item['target_path']).exists())

    def test_existing_good_file_needs_no_network(self):
        target = self.root / self.item['target_path']
        target.parent.mkdir()
        target.write_bytes(self.data)
        with patch.object(m.urllib.request, 'urlopen') as call:
            m.ensure_file(self.root, self.manifest, self.item)
            call.assert_not_called()

    def test_verify_only_never_downloads(self):
        with patch.object(m.urllib.request, 'urlopen') as call:
            with self.assertRaises(FileNotFoundError):
                m.ensure_file(self.root, self.manifest, self.item, verify_only=True)
            call.assert_not_called()

    def test_symlink_escape_is_rejected(self):
        (self.root / 'escape').symlink_to('/tmp', target_is_directory=True)
        with self.assertRaises(ValueError):
            m.safe_target(self.root, 'escape/weight.gguf')

    def test_manifest_traversal_is_rejected(self):
        data = json.loads((ROOT / 'model-manifest.json').read_text())
        data['files'][0]['target_path'] = '../escaped.gguf'
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            m.load_manifest(path)


if __name__ == '__main__':
    unittest.main()
