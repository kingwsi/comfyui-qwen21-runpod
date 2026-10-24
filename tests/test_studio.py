import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / 'studio'
if not STUDIO.is_dir():
    STUDIO = Path('/opt/ComfyUI/custom_nodes/qwen_studio')
# Import pure helpers without executing ComfyUI's registration entry point.
package = types.ModuleType('studio_test_package')
package.__path__ = [str(STUDIO)]
sys.modules.setdefault('studio_test_package', package)
graph = importlib.import_module('studio_test_package.graph')
Store = importlib.import_module('studio_test_package.store').Store


class StudioGraphs(unittest.TestCase):
    def test_all_templates_and_both_model_profiles(self):
        for variant in ('base', 'turbo'):
            for template, spec in graph.TEMPLATES.items():
                if template not in graph.templates_for_variant(variant):
                    continue
                for count in set((spec['min_images'], spec['max_images'])):
                    prompt = graph.template_prompt(template, '只修改指定内容', count)
                    images = [f'qwen_studio/test_{i}.png' for i in range(count)]
                    result = graph.build_graph(prompt, images, variant, seed=123, template=template)
                    self.assertEqual(result['4']['inputs']['prompt'], prompt)
                    refs = {k: v for k,v in result['4']['inputs'].items() if k.startswith('images.')}
                    self.assertEqual(len(refs), count)
                    self.assertEqual(result['4']['inputs']['resolution'], 0 if count else 1024)
                    self.assertEqual(result['6']['inputs']['latent_image'], ['4', 2])
                    for i, name in enumerate(images, 1):
                        scale = result[refs[f'images.image_{i}'][0]]
                        self.assertEqual(scale['inputs']['largest_size'], 1024)
                        self.assertEqual(result[scale['inputs']['image'][0]]['inputs']['image'], name)
                    self.assertEqual(result['9']['class_type'], 'SaveImage')
                    if variant == 'turbo':
                        self.assertEqual(result['11']['inputs']['sigmas'], graph.SIGMAS)
                        self.assertEqual(result['6']['class_type'], 'SamplerCustom')
                        self.assertEqual(result['6']['inputs']['noise_seed'], 123)
                    else:
                        self.assertEqual(result['6']['inputs']['steps'], 25)

    def test_bfs_is_turbo_only_and_applies_before_cache(self):
        self.assertNotIn('faceswap', graph.templates_for_variant('base'))
        self.assertIn('faceswap', graph.templates_for_variant('turbo'))
        images = ['qwen_studio/target.png', 'qwen_studio/head.png']
        result = graph.build_graph(graph.template_prompt('faceswap', '保持表情', 2), images, 'turbo', seed=123, template='faceswap')
        self.assertEqual(result['12']['class_type'], 'LoraLoaderModelOnly')
        self.assertEqual(result['12']['inputs'], dict(model=['1',0], lora_name=graph.BFS_FILENAME, strength_model=1.0))
        self.assertEqual(result['5']['inputs']['model'], ['12',0])
        self.assertTrue(result['4']['inputs']['prompt'].startswith('head_swap:'))
        self.assertIn('<image1>', result['4']['inputs']['prompt'])
        self.assertIn('<image2>', result['4']['inputs']['prompt'])
        for variant, count in [('base',2),('turbo',1),('turbo',3)]:
            with self.assertRaises(ValueError):
                graph.build_graph('test', ['input.png']*count, variant, template='faceswap')
        for template in ('text','edit','reference'):
            result = graph.build_graph('test', images, 'turbo', template=template)
            self.assertNotIn('LoraLoaderModelOnly', [n['class_type'] for n in result.values()])

    def test_independent_prompts_and_template_limits(self):
        first = graph.template_prompt('edit', '改成暖色', 1)
        second = graph.template_prompt('edit', '改成冷色', 1)
        self.assertNotIn('暖色', second)
        self.assertIn('冷色', second)
        self.assertIn('暖色', first)
        for args in [('text','x',1),('edit','x',0),('reference','x',1),('reference','x',4),('background','x',2),('other','x',0),('text','',0),('text','x'*4001,0)]:
            with self.assertRaises(ValueError):
                graph.template_prompt(*args)

    def test_packaged_without_extra_public_port(self):
        self.assertIn('COPY studio/ /opt/ComfyUI/custom_nodes/qwen_studio/', (ROOT/'Dockerfile').read_text())
        for pattern in ('!studio/*.py', '!studio/static/*.js', '!studio/static/*.css', '!studio/static/*.html'):
            self.assertIn(pattern, (ROOT/'.dockerignore').read_text())
        html = (STUDIO/'static/index.html').read_text()
        self.assertNotIn('https://', html)
        self.assertNotIn('对话', html)


class StudioStore(unittest.TestCase):
    def test_restart_persistence_and_owner_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            job = dict(id='job', status='queued', template='edit', prompt='只改背景', assets=[])
            store.save('alice', job)
            aid = store.add_asset('alice', '/tmp/example.png', 'qwen_studio/example.png')
            store = Store(root)
            self.assertEqual(store.get('alice','job')['prompt'], '只改背景')
            self.assertEqual(store.latest('alice'), 'job')
            self.assertIsNone(store.latest('bob'))
            for method, arg in ((store.get,'job'),(store.asset,aid)):
                with self.assertRaises(KeyError): method('bob', arg)
            self.assertEqual(store.asset('alice',aid)['input_name'], 'qwen_studio/example.png')
