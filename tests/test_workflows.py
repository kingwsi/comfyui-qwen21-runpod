import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Workflows(unittest.TestCase):
    def test_workflows_are_source_only_and_linked(self):
        files=list((ROOT/'workflows').glob('*.json'))
        self.assertEqual(len(files),4)
        for p in files:
            data=json.loads(p.read_text())
            nodes={n['id']:n for n in data['nodes']}
            self.assertEqual(len(nodes),len(data['nodes']))
            self.assertEqual(data['version'],0.4)
            links={x[0]:x for x in data['links']}
            self.assertEqual(len(links),len(data['links']))
            for link in links.values():
                lid,source,outslot,target,inslot,kind=link
                self.assertIn(source,nodes);self.assertIn(target,nodes)
                self.assertEqual(nodes[target]['inputs'][inslot]['link'],lid)
                self.assertIn(lid,nodes[source]['outputs'][outslot]['links'])
            for node in nodes.values():
                if node['type']=='LoadImage':
                    self.assertIn(node['widgets_values'][0],['person.png','clothing.png','original.png','main.png','reference_1.png','reference_2.png'])
                for slot in node.get('inputs',[]):
                    if slot.get('link') is not None:self.assertIn(slot['link'],links)
                for slot in node.get('outputs',[]):
                    for lid in slot.get('links') or []:self.assertIn(lid,links)
            self.assertNotIn('data:image',p.read_text())

    def test_1024_sizing_and_optional_lora(self):
        for p in (ROOT/'workflows').glob('*.json'):
            self.assertIn('1024',p.name)
            d=json.loads(p.read_text()); types=[n['type'] for n in d['nodes']]
            for n in d['nodes']:
                if n['type']=='ImageScaleToMaxDimension':self.assertEqual(n['widgets_values'],['lanczos',1024])
                if n['type']=='ImageScale':self.assertEqual(n['widgets_values'],['lanczos',1024,1024,'center'])
            self.assertNotIn('ImageScaleToTotalPixels',types)
            loras=[n for n in d['nodes'] if n['type']=='LoraLoaderModelOnly']
            if 'Outfit' in p.name:
                self.assertEqual(len(loras),1)
                self.assertEqual(loras[0]['widgets_values'],['OutfitSwap-LoRA-GGUF-compatible.safetensors',1.0])
            else:self.assertEqual(loras,[])
            expected=3 if '3Images' in p.name else 2 if ('2Images' in p.name or 'Outfit' in p.name) else 1
            self.assertEqual(types.count('LoadImage'),expected)

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
