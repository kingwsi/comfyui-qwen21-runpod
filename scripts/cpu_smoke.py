"""CPU-only container smoke: no weights/network/GPU inference required."""
import json, os, secrets, subprocess, tempfile, time, urllib.request, urllib.error, http.cookiejar, sys
from pathlib import Path

def combo_options(spec):
    # Legacy nodes return [options, metadata]; V3 returns ["COMBO", metadata].
    if isinstance(spec[0], (list, tuple)):
        return spec[0]
    if spec[0] == 'COMBO' and len(spec) > 1:
        options=spec[1].get('options')
        if isinstance(options, (list, tuple)):
            return options
    raise AssertionError('Unsupported runtime combo schema')

def check_turbo_runtime(info):
    # Check real object_info, not only whether a node class was registered.
    contracts={
        'CLIPLoaderGGUF': ({'clip_name','type'}, ['CLIP']),
        'SamplerCustom': ({'model','add_noise','noise_seed','cfg','positive','negative','sampler','sigmas','latent_image'}, ['LATENT','LATENT']),
        'ManualSigmas': ({'sigmas'}, ['SIGMAS']),
        'KSamplerSelect': ({'sampler_name'}, ['SAMPLER']),
    }
    for name,(required,outputs) in contracts.items():
        assert name in info, ('Missing Turbo node',name)
        actual=info[name]
        assert required.issubset(actual['input']['required']), ('Turbo input mismatch',name)
        assert list(actual['output'])==outputs, ('Turbo output mismatch',name)
    assert 'qwen_image' in combo_options(info['CLIPLoaderGGUF']['input']['required']['type'])
    assert 'euler' in combo_options(info['KSamplerSelect']['input']['required']['sampler_name'])

def check_studio_runtime(info):
    sys.path.insert(0, '/opt/comfyui-reusable/tests')
    from test_studio import graph
    for variant in ('base', 'turbo'):
        for count, template in [(i, None) for i in range(4)] + ([(2, 'faceswap')] if variant == 'turbo' else []):
            prompt = graph.build_graph('smoke test', [f'test-{i}.png' for i in range(count)], variant, seed=1, template=template)
            for node in prompt.values():
                schema = info[node['class_type']]
                inputs = dict(schema['input'].get('required', {}), **schema['input'].get('optional', {}))
                for name, value in node['inputs'].items():
                    assert name in inputs or (name.startswith('images.image_') and 'images' in inputs), (node['class_type'], name)
                    if isinstance(value, list):
                        source = prompt[value[0]]
                        assert value[1] < len(info[source['class_type']]['output'])
    print('PASS: studio API graph runtime contracts, both profiles, text/single/2/3 images.')


def get(url, token=None, data=None, method=None):
    headers = {'Authorization': 'token ' + token} if token else {}
    if data is not None: headers['Content-Type'] = 'application/json'
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=10) as r:
        return r.status, r.read()

def main():
    import websocket
    import fastjsonschema
    token = secrets.token_hex(24)
    procs=[]
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        for name in ("user", "input", "output", "temp"):
            (root/name).mkdir()
        try:
            logs=[]
            cmds=[
                ['python','/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1','--port','8188','--disable-auto-launch','--user-directory',str(root/'user'),'--output-directory',str(root/'output'),'--input-directory',str(root/'input'),'--temp-directory',str(root/'temp')],
                ['python','-m','jupyter','server','--allow-root','--no-browser','--ServerApp.ip=127.0.0.1','--ServerApp.port=8888','--IdentityProvider.token='+token,'--ServerApp.root_dir='+str(root)],
            ]
            for i,cmd in enumerate(cmds):
                log=open(root/f'process-{i}.log','w+'); logs.append(log)
                procs.append(subprocess.Popen(cmd,cwd='/opt/ComfyUI',stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'RUNTIME_DIR':str(root)}))
            base='http://127.0.0.1:8888/proxy/8188/'
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if any(p.poll() is not None for p in procs): raise RuntimeError('Service exited during startup')
                try:
                    # Wait for ComfyUI itself before probing the proxy. Otherwise
                    # normal startup delay produces misleading proxy 500 traces.
                    get('http://127.0.0.1:8188/object_info')
                    status, body=get(base+'object_info',token)
                    info=json.loads(body)
                    assert 'UnetLoaderGGUF' in info
                    break
                except (OSError,ValueError,AssertionError): time.sleep(2)
            else: raise RuntimeError('CPU ComfyUI/proxy startup timed out')
            check_turbo_runtime(info)
            check_studio_runtime(info)
            # Exercise the shipped custom routes through the real authenticated
            # proxy, including cookie-based asset isolation and relative paths.
            opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            headers={'Authorization':'token '+token}
            with opener.open(urllib.request.Request(base+'qwen-studio/',headers=headers),timeout=10) as response:
                assert response.status==200 and '生成模板'.encode() in response.read()
            with opener.open(urllib.request.Request(base+'qwen-studio/api/config',headers=headers),timeout=10) as response:
                config=json.load(response)
                expected_templates={'text','edit','reference'}
                if config['model']=='turbo': expected_templates.add('faceswap')
                assert set(config['templates'])==expected_templates
            for name in ('app.js','style.css'):
                assert get(base+'qwen-studio/'+name,token)[0]==200
            try:
                _,body=get(base+'qwen-studio/api/config')
                assert b'"templates"' not in body, 'Anonymous access leaked studio config'
            except urllib.error.HTTPError as e:
                assert e.code in (401,403)
            print('PASS: studio page/assets/config through authenticated Jupyter; anonymous access denied.')
            validate=fastjsonschema.compile(json.loads(Path('/opt/comfyui-reusable/tests/comfy-workflow-0.4.schema.json').read_text()))
            # Every shipped graph must be valid and its nodes registered.
            count=0
            for path in Path('/opt/comfyui-reusable/workflows').glob('*.json'):
                workflow=json.loads(path.read_text())
                validate(workflow)
                count+=1
                for node in workflow['nodes']:
                    if node['type'] != 'Note':
                        assert node['type'] in info, (path.name, node['type'])
            assert count==6, ('Unexpected workflow count',count)
            print(f'PASS: all {count} workflow schemas and node registrations; Turbo runtime input/output contracts.')
            # Anonymous request must not reach ComfyUI object_info JSON.
            try:
                _,body=get(base+'object_info')
                assert b'UnetLoaderGGUF' not in body, 'Anonymous proxy access leaked ComfyUI'
            except urllib.error.HTTPError as e:
                assert e.code in (401,403)
            path='userdata/workflows%2Fci-smoke.json'
            payload=b'{"source":"cpu-smoke"}'
            assert get(base+path,token,payload,'POST')[0] in (200,201,204)
            assert json.loads(get(base+path,token)[1]) == json.loads(payload)
            ws=websocket.create_connection('ws://127.0.0.1:8888/proxy/8188/ws?clientId=ci-smoke',header=['Authorization: token '+token],origin='http://127.0.0.1:8888',timeout=15)
            message=json.loads(ws.recv()); ws.close()
            assert message['type']=='status'
            print('PASS: CPU startup, GGUF node registration, authenticated HTTP, encoded-slash userdata round-trip, WebSocket and anonymous denial. No GPU inference.')
        except Exception:
            for log in logs:
                log.flush(); log.seek(0)
                print(log.read().replace(token,'[redacted]')[-12000:])
            raise
        finally:
            for p in procs: p.terminate()
            for p in procs:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
if __name__=='__main__':main()
