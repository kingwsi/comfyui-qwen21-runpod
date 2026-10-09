import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Workflows(unittest.TestCase):
    def test_workflows_are_source_only_and_linked(self):
        files=list((ROOT/'workflows').glob('*.json'))
        self.assertEqual(len(files),2)
        for p in files:
            data=json.loads(p.read_text())
            nodes={n['id']:n for n in data['nodes']}
            self.assertEqual(len(nodes),len(data['nodes']))
            self.assertEqual(data['version'],0.4)
            for link in data['links']:
                _,source,_,target,_,_=link
                self.assertIn(source,nodes);self.assertIn(target,nodes)
            for node in nodes.values():
                if node['type']=='LoadImage':
                    self.assertIn(node['widgets_values'][0],['person.png','clothing.png','original.png'])
            self.assertNotIn('data:image',p.read_text())
    def test_locked_no_cuda_reinstall(self):
        names=[]
        for line in (ROOT/'payload-requirements.txt').read_text().splitlines():
            if not line or line.startswith('#'):continue
            name,version=line.split('==');self.assertTrue(version)
            names.append(name.lower().replace('_','-'))
        self.assertEqual(len(names),len(set(names)))
        self.assertFalse(any(n in ('torch','torchvision','torchaudio','triton') or n.startswith('nvidia-') for n in names))
    def test_no_paid_runner_or_token_persistence(self):
        text=(ROOT/'.github/workflows/build-image.yml').read_text()
        self.assertIn('runs-on: ubuntu-24.04',text)
        self.assertIn('persist-credentials: false',text)
        self.assertIn('secrets.GITHUB_TOKEN',text)
        self.assertNotIn('self-hosted',text)
if __name__=='__main__':unittest.main()
