import asyncio
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import secrets
import uuid
import weakref
from aiohttp import ClientError, ClientSession, ClientTimeout, web
from PIL import Image, ImageOps, UnidentifiedImageError
from .graph import build_graph, template_prompt, templates_for_variant
from .store import Store

LOG = logging.getLogger(__name__)
COOKIE = 'qwen_studio_session'
BASE = '/qwen-studio'
ACTIVE = {'submitting', 'queued', 'running'}
STATIC = Path(__file__).parent / 'static'
UPLOAD_LIMIT = 20 * 1024 * 1024


class StudioAPI:
    def __init__(self, folders, upstream='http://127.0.0.1:8188', root=None):
        self.folders = folders
        self.upstream = upstream
        self.store = Store(root or Path(os.environ.get('RUNTIME_DIR', '/workspace/comfyui-data')) / 'studio')
        self.locks = weakref.WeakValueDictionary()
        self.variant = os.environ.get('MODEL_VARIANT', 'base')

    def owner(self, request):
        owner = request.cookies.get(COOKIE, '')
        if len(owner) != 64 or any(c not in '0123456789abcdef' for c in owner):
            raise web.HTTPUnauthorized(text='请刷新页面后重试。')
        return owner

    def write_access(self, request):
        # The custom header cannot be submitted by a cross-origin HTML form.
        # Sec-Fetch-Site also works behind Jupyter's rewritten upstream Host.
        if request.headers.get('X-Qwen-Studio') != '1' or request.headers.get('Sec-Fetch-Site') not in (None, 'same-origin', 'none'):
            raise web.HTTPForbidden(text='不允许跨站请求。')
        return self.owner(request)

    async def call(self, path, payload=None):
        async with ClientSession(timeout=ClientTimeout(total=30)) as client:
            async with client.request('POST' if payload is not None else 'GET', self.upstream + path, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    error = data.get('error', {})
                    message = error.get('message') if isinstance(error, dict) else str(error)
                    raise ValueError('ComfyUI 拒绝任务：' + (message or str(data.get('node_errors') or response.status))[:1000])
                return data

    async def redirect(self, request):
        # Relative Location retains /proxy/8188/ through the Jupyter proxy.
        raise web.HTTPFound(location='qwen-studio/')

    async def index(self, request):
        response = web.FileResponse(STATIC / 'index.html')
        owner = request.cookies.get(COOKIE, '')
        if len(owner) != 64 or any(c not in '0123456789abcdef' for c in owner):
            response.set_cookie(COOKIE, secrets.token_hex(32), httponly=True, samesite='Strict', max_age=31536000, path='/')
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'"
        return response

    async def static(self, request):
        name = request.match_info['name']
        if name not in ('app.js', 'style.css'):
            raise web.HTTPNotFound()
        response = web.FileResponse(STATIC / name)
        response.headers['Cache-Control'] = 'no-cache'
        return response

    async def config(self, request):
        owner = self.owner(request)
        return web.json_response(dict(templates=templates_for_variant(self.variant), model=self.variant, latest_job=self.store.latest(owner)))

    def lock(self, jid):
        lock = self.locks.get(jid)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[jid] = lock
        return lock

    async def reconcile(self, owner, job):
        if job['status'] not in ACTIVE:
            return
        pid = job.get('prompt_id')
        if not pid:
            # A submitting lock cannot survive a restart; never auto-resubmit
            # an ambiguous job and inadvertently run it twice.
            job.update(status='failed', error='服务在提交时中断，请检查 ComfyUI 队列后重新生成。')
        else:
            history = (await self.call('/history/' + pid)).get(pid)
            if history:
                if history.get('status', {}).get('status_str') == 'error':
                    messages = history.get('status', {}).get('messages', [])
                    errors = [m[1].get('exception_message', '') for m in messages if m[0] == 'execution_error']
                    job.update(status='failed', error=('生成失败：' + ' '.join(errors))[:1500])
                elif history.get('status', {}).get('completed'):
                    images = history.get('outputs', {}).get('9', {}).get('images', [])
                    if not images:
                        job.update(status='failed', error='ComfyUI 已结束，但未返回保存图片。')
                    else:
                        image = images[0]
                        root = Path(self.folders.get_output_directory()).resolve()
                        path = (root / image.get('subfolder', '') / image['filename']).resolve()
                        if image.get('type') != 'output' or not path.is_relative_to(root) or not path.is_file():
                            raise ValueError('ComfyUI 返回了无效的图片路径。')
                        aid = self.store.add_asset(owner, path)
                        job.update(status='done', result=aid)
            else:
                queue = await self.call('/queue')
                running = [x[1] for x in queue.get('queue_running', [])]
                waiting = [x[1] for x in queue.get('queue_pending', [])]
                if pid in running:
                    job.update(status='running', missing=0)
                elif pid in waiting:
                    job.update(status='queued', missing=0, position=waiting.index(pid) + 1)
                else:
                    # History and queue reads may straddle job completion.
                    job['missing'] = job.get('missing', 0) + 1
                    if job['missing'] >= 3:
                        job.update(status='failed', error='任务已不在队列中，可能因服务重启或队列清理而中断。')
        self.store.save(owner, job)

    async def detail(self, request):
        owner, jid = self.owner(request), request.match_info['jid']
        async with self.lock(jid):
            job = self.store.get(owner, jid)
            await self.reconcile(owner, job)
            return web.json_response(job)

    async def generate(self, request):
        owner = self.write_access(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('请求格式无效。')
        rid = body.get('request_id', '')
        try:
            rid = uuid.UUID(rid).hex
        except (ValueError, TypeError, AttributeError):
            raise ValueError('缺少有效的请求标识。')
        jid = hashlib.sha256((owner + rid).encode()).hexdigest()[:32]
        async with self.lock(jid):
            try:
                return web.json_response(self.store.get(owner, jid))
            except KeyError:
                pass
            template, text, assets = body.get('template'), body.get('prompt'), body.get('assets', [])
            if not isinstance(template, str) or not isinstance(assets, list) or any(not isinstance(a, str) for a in assets):
                raise ValueError('模板或图片格式无效。')
            if len(assets) != len(set(assets)):
                raise ValueError('请勿重复选择同一张图片。')
            if template not in templates_for_variant(self.variant):
                raise ValueError('当前模型不支持此模板；换脸 / 换头需要 MODEL_VARIANT=turbo。')
            prompt = template_prompt(template, text, len(assets))
            images = [self.store.asset(owner, aid)['input_name'] for aid in assets]
            if any(not name for name in images):
                raise ValueError('该图片不能用作输入。')
            graph = build_graph(prompt, images, self.variant, template=template)
            job = dict(id=jid, template=template, prompt=text.strip(), assets=assets, status='submitting',
                       seed=graph['6']['inputs'].get('seed', graph['6']['inputs'].get('noise_seed')))
            self.store.save(owner, job)
            try:
                result = await self.call('/prompt', dict(prompt=graph, client_id='qwen-studio-' + jid))
                job.update(status='queued', prompt_id=result['prompt_id'])
            except (ClientError, asyncio.TimeoutError, OSError):
                job.update(status='failed', error='提交时连接中断，无法确认任务是否已入队。请先查看 ComfyUI 队列，再决定是否重新生成。')
            except Exception as exc:
                LOG.warning('Qwen studio submission failed: %s', type(exc).__name__)
                job.update(status='failed', error=str(exc)[:1500] or '提交失败，请稍后重试。')
            self.store.save(owner, job)
            return web.json_response(job)

    async def upload(self, request):
        owner = self.write_access(request)
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != 'image':
            raise ValueError('请选择图片。')
        raw = bytearray()
        while chunk := await part.read_chunk():
            raw.extend(chunk)
            if len(raw) > UPLOAD_LIMIT:
                raise ValueError('每张图片不能超过 20 MB。')
        name = 'qwen_studio/' + uuid.uuid4().hex + '.png'
        path = Path(self.folders.get_input_directory()) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        def normalize():
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in ('PNG', 'JPEG', 'WEBP') or image.width * image.height > 40_000_000:
                        raise ValueError('请使用不超过 4000 万像素的 PNG、JPEG 或 WebP 图片。')
                    image.load()
                    ImageOps.exif_transpose(image).convert('RGBA').save(path, format='PNG')
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError('图片无法读取，请换一张图片。') from exc
        await asyncio.to_thread(normalize)
        aid = self.store.add_asset(owner, path, name)
        return web.json_response(dict(id=aid))

    async def asset(self, request):
        asset = self.store.asset(self.owner(request), request.match_info['aid'])
        response = web.FileResponse(asset['path'])
        response.headers['Cache-Control'] = 'private, no-cache'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if request.query.get('download') == '1':
            response.headers['Content-Disposition'] = 'attachment; filename="qwen-image.png"'
        return response


def register(routes, folders, **kwargs):
    api = StudioAPI(folders, **kwargs)
    def endpoint(handler):
        async def guarded(request):
            try:
                response = await handler(request)
                response.headers.setdefault('Cache-Control', 'no-store')
                return response
            except KeyError:
                return web.json_response({'error': '任务或图片不存在。'}, status=404)
            except ValueError as exc:
                return web.json_response({'error': str(exc)}, status=400)
            except web.HTTPException as exc:
                if 300 <= exc.status < 400:
                    raise
                return web.json_response({'error': exc.text}, status=exc.status)
            except Exception:
                LOG.exception('Qwen studio request failed')
                return web.json_response({'error': '暂时无法连接 ComfyUI 或读取文件，请稍后重试并查看日志。'}, status=503)
        return guarded
    routes.get(BASE)(endpoint(api.redirect))
    routes.get(BASE + '/')(endpoint(api.index))
    routes.get(BASE + '/{name:app.js|style.css}')(endpoint(api.static))
    routes.get(BASE + '/api/config')(endpoint(api.config))
    routes.post(BASE + '/api/jobs')(endpoint(api.generate))
    routes.get(BASE + '/api/jobs/{jid}')(endpoint(api.detail))
    routes.post(BASE + '/api/uploads')(endpoint(api.upload))
    routes.get(BASE + '/api/assets/{aid}')(endpoint(api.asset))
    return api
