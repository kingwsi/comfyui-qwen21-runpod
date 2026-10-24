import asyncio
import importlib
import io
import tempfile
import unittest
import uuid
from pathlib import Path
from test_studio import STUDIO

try:
    from aiohttp import FormData, web
    from aiohttp.test_utils import TestClient, TestServer
    from PIL import Image
    register = importlib.import_module('studio_test_package.api').register
    AVAILABLE = True
except ImportError:
    if STUDIO.parent.name == 'custom_nodes':
        raise  # Runtime image must run the HTTP integration tests.
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'aiohttp/Pillow available in the image; source-only environment may skip HTTP tests')
class StudioAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'input').mkdir(); (self.root/'output').mkdir()
        root = self.root
        class Folders:
            def get_input_directory(self): return str(root/'input')
            def get_output_directory(self): return str(root/'output')
        self.folders = Folders()
        self.graphs = []; self.history = {}; self.pending = []; self.running = []
        self.reject = False
        async def prompt(request):
            body = await request.json()
            if self.reject: return web.json_response({'error': {'message':'model missing'}},status=400)
            pid = str(uuid.uuid4()); self.graphs.append(body['prompt']); self.pending.append([len(self.graphs),pid])
            await asyncio.sleep(.01)
            return web.json_response({'prompt_id':pid})
        async def history(request): return web.json_response(self.history)
        async def queue(request): return web.json_response({'queue_pending':self.pending,'queue_running':self.running})
        upstream = web.Application()
        upstream.add_routes([web.post('/prompt',prompt),web.get('/history/{pid}',history),web.get('/queue',queue)])
        self.upstream = TestServer(upstream); await self.upstream.start_server()
        self.routes = web.RouteTableDef()
        self.api = register(self.routes, self.folders, upstream=str(self.upstream.make_url('')).rstrip('/'), root=self.root/'data')
        app = web.Application(); app.add_routes(self.routes)
        self.client = TestClient(TestServer(app)); await self.client.start_server()
        response = await self.client.get('/qwen-studio/')
        self.assertEqual(response.status,200); await response.read()
        self.headers = {'X-Qwen-Studio':'1'}

    async def asyncTearDown(self):
        await self.client.close(); await self.upstream.close(); self.temp.cleanup()

    async def upload(self, raw=None):
        if raw is None:
            output=io.BytesIO(); Image.new('RGB',(48,32),'green').save(output,format='PNG'); raw=output.getvalue()
        form=FormData(); form.add_field('image', raw, filename='image.png', content_type='image/png')
        response=await self.client.post('/qwen-studio/api/uploads',data=form,headers=self.headers)
        return response,await response.json()

    async def submit(self, template='text', assets=None, prompt='画一片海', rid=None):
        return await self.client.post('/qwen-studio/api/jobs',json=dict(template=template,assets=assets or [],prompt=prompt,request_id=rid or str(uuid.uuid4())),headers=self.headers)

    async def finish(self, job):
        name='result.png'; Image.new('RGB',(64,64),'blue').save(self.root/'output'/name)
        self.pending=[]
        self.history[job['prompt_id']]={'status':{'completed':True,'status_str':'success'},'outputs':{'9':{'images':[{'filename':name,'subfolder':'','type':'output'}]}}}
        response=await self.client.get('/qwen-studio/api/jobs/'+job['id'])
        return await response.json()

    async def test_upload_generate_all_templates_and_independent_jobs(self):
        for variant in ('base','turbo'):
            self.api.variant=variant
            cases = [('text',0),('edit',1),('reference',2),('reference',3)]
            if variant == 'turbo': cases.append(('faceswap',2))
            for template,count in cases:
                assets=[]
                for _ in range(count):
                    response,body=await self.upload(); self.assertEqual(response.status,200); assets.append(body['id'])
                response=await self.submit(template,assets)
                self.assertEqual(response.status,200)
                job=await response.json(); self.assertEqual(job['status'],'queued')
                result=await self.finish(job); self.assertEqual(result['status'],'done')
                image=await self.client.get('/qwen-studio/api/assets/'+result['result']+'?download=1')
                self.assertEqual(image.status,200); self.assertIn('attachment',image.headers['Content-Disposition']); await image.read()
                self.assertEqual(result['assets'],assets)
                self.assertEqual(len([n for n in self.graphs[-1].values() if n['class_type']=='LoadImage']),count)
                loras = [n for n in self.graphs[-1].values() if n['class_type']=='LoraLoaderModelOnly']
                self.assertEqual(len(loras), 1 if template == 'faceswap' else 0)
                if loras: self.assertEqual(loras[0]['inputs']['strength_model'], 1.0)
        self.assertEqual(len(self.graphs),9)
        # A new request neither inherits the previous prompt nor its images.
        response=await self.submit(prompt='画一朵花'); await response.read()
        self.assertNotIn('海',self.graphs[-1]['4']['inputs']['prompt'])
        self.assertFalse(any(n['class_type']=='LoadImage' for n in self.graphs[-1].values()))

    async def test_bfs_config_and_non_turbo_requests_are_rejected(self):
        assets = [(await self.upload())[1]['id'] for _ in range(2)]
        self.api.variant = 'base'
        response = await self.client.get('/qwen-studio/api/config')
        self.assertNotIn('faceswap', (await response.json())['templates'])
        response = await self.submit('faceswap', assets)
        self.assertEqual(response.status, 400)
        self.assertEqual(self.graphs, [])
        self.api.variant = 'turbo'
        response = await self.client.get('/qwen-studio/api/config')
        template = (await response.json())['templates']['faceswap']
        self.assertEqual(template['image_labels'], ['目标图片', '参考头像'])
        for count in (1,3):
            response = await self.submit('faceswap', (assets + ['other'])[:count])
            self.assertEqual(response.status,400)
        self.assertEqual(self.graphs, [])

    async def test_concurrent_retry_is_idempotent(self):
        rid=str(uuid.uuid4())
        responses=await asyncio.gather(self.submit(rid=rid),self.submit(rid=rid))
        jobs=[await r.json() for r in responses]
        self.assertEqual(jobs[0]['id'],jobs[1]['id']); self.assertEqual(len(self.graphs),1)
        response=await self.client.get('/qwen-studio/api/jobs/'+jobs[0]['id']); queued=await response.json()
        self.assertEqual(queued['position'],1)
        self.running=self.pending; self.pending=[]
        response=await self.client.get('/qwen-studio/api/jobs/'+jobs[0]['id']); self.assertEqual((await response.json())['status'],'running')

    async def test_failures_and_restart_recovery(self):
        self.reject=True
        response=await self.submit(); failed=await response.json()
        self.assertEqual(failed['status'],'failed'); self.assertIn('model missing',failed['error'])
        self.reject=False
        response=await self.submit(); job=await response.json()
        self.history[job['prompt_id']]={'status':{'completed':False,'status_str':'error','messages':[['execution_error',{'exception_message':'out of memory'}]]}}
        response=await self.client.get('/qwen-studio/api/jobs/'+job['id']); self.assertIn('out of memory',(await response.json())['error'])
        response=await self.submit(); job=await response.json()
        # Reload the store, simulating a process restart with its queue preserved.
        from studio_test_package.store import Store
        self.api.store=Store(self.root/'data')
        response=await self.client.get('/qwen-studio/api/config'); self.assertEqual((await response.json())['latest_job'],job['id'])
        result=await self.finish(job); self.assertEqual(result['status'],'done')
        self.assertEqual(len(self.graphs),2)

    async def test_owner_isolation_csrf_and_bad_inputs(self):
        response,asset=await self.upload(); self.assertEqual(response.status,200)
        response,body=await self.upload(b'not an image'); self.assertEqual(response.status,400)
        response=await self.submit('reference',[asset['id']]); self.assertEqual(response.status,400)
        response=await self.submit('text',[asset['id']]); self.assertEqual(response.status,400)
        response=await self.client.post('/qwen-studio/api/jobs',json={}); self.assertEqual(response.status,403)
        response=await self.client.post('/qwen-studio/api/jobs',json={},headers={**self.headers,'Sec-Fetch-Site':'cross-site'}); self.assertEqual(response.status,403)
        response=await self.submit(); job=await response.json()
        self.client.session.cookie_jar.clear()
        response=await self.client.get('/qwen-studio/api/config'); self.assertEqual(response.status,401)
        await (await self.client.get('/qwen-studio/')).read()
        for url in ('/qwen-studio/api/jobs/'+job['id'],'/qwen-studio/api/assets/'+asset['id']):
            response=await self.client.get(url); self.assertEqual(response.status,404)
        response=await self.submit('edit',[asset['id']]); self.assertEqual(response.status,404)
        self.assertEqual(len(self.graphs),1)

    async def test_output_traversal_is_refused_and_missing_queue_is_reported(self):
        response=await self.submit(); job=await response.json()
        self.history[job['prompt_id']]={'status':{'completed':True},'outputs':{'9':{'images':[{'filename':'../outside.png','type':'output'}]}}}
        response=await self.client.get('/qwen-studio/api/jobs/'+job['id']); self.assertEqual(response.status,400)
        self.history={}; self.pending=[]
        for _ in range(3):
            response=await self.client.get('/qwen-studio/api/jobs/'+job['id']); final=await response.json()
        self.assertEqual(final['status'],'failed')
