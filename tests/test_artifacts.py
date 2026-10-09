import json,re,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Artifacts(unittest.TestCase):
 def test_base_digest_pinned(self):
  s=json.loads((ROOT/'sources.json').read_text());self.assertRegex(s['base_image_digest'],r'^sha256:[0-9a-f]{64}$');self.assertIn(s['base_image_digest'],(ROOT/'Dockerfile').read_text())
 def test_inherits_startup(self):
  lines=[x for x in (ROOT/'Dockerfile').read_text().splitlines() if not x.startswith('#')];self.assertFalse(any(re.match(r'^(ENTRYPOINT|CMD)\s',x) for x in lines));self.assertIn('test ! -e /pre_start.sh','\n'.join(lines))
 def test_lora_pinned(self):
  s=json.loads((ROOT/'lora-manifest.json').read_text());self.assertRegex(s['revision'],r'^[0-9a-f]{40}$');self.assertEqual(s['files'][0]['sha256'],'4ddfeac5695adaed1f0f78b23fda40919fd2ed6a4af790ced6c1e043dc4ae1f0');self.assertEqual(s['total_size_bytes'],159436488)
 def test_no_cpu_torch_payload(self):
  self.assertFalse((ROOT/'site-packages/torch').exists());self.assertFalse((ROOT/'site-packages/torchvision').exists())
 def test_auth_and_bind(self):
  self.assertIn('JUPYTER_PASSWORD must be configured',(ROOT/'pre_start.sh').read_text());self.assertIn('--listen 127.0.0.1',(ROOT/'start.sh').read_text());self.assertNotIn('disable_check_xsrf',(ROOT/'Dockerfile').read_text())
 def test_jupyter_cli_uses_venv(self):
  self.assertIn('--no-deps --ignore-installed -r',(ROOT/'Dockerfile').read_text())
  self.assertIn('(root/name).mkdir()',(ROOT/'scripts/cpu_smoke.py').read_text())
 def test_converter_and_tests_are_packaged(self):
  docker=(ROOT/'Dockerfile').read_text()
  self.assertIn('COPY scripts/ /opt/comfyui-reusable/scripts/',docker)
  self.assertIn('COPY tests/ /opt/comfyui-reusable/tests/',docker)
  self.assertIn('COPY Dockerfile README.md pre_start.sh /opt/comfyui-reusable/',docker)
  ignore=(ROOT/'.dockerignore').read_text()
  for pattern in ('!scripts/*.py','!scripts/*.sh','!tests/*.py','!README.md','!.github/workflows/*.yml'):
   self.assertIn(pattern,ignore)
  workflow=(ROOT/'.github/workflows/build-image.yml').read_text()
  self.assertIn('-m unittest discover -s /opt/comfyui-reusable/tests -v',workflow)
 def test_proxy_patch_present(self):
  self.assertIn('self.request.path[len(raw_prefix):]',(ROOT/'scripts/patch_proxy.py').read_text())
if __name__=='__main__':unittest.main()
