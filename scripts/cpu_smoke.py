"""CPU-only container smoke: no weights/network/GPU inference required."""
import json, os, secrets, subprocess, tempfile, time, urllib.request, urllib.error
from pathlib import Path
import websocket

def get(url, token=None, data=None, method=None):
    headers = {'Authorization': 'token ' + token} if token else {}
    if data is not None: headers['Content-Type'] = 'application/json'
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=10) as r:
        return r.status, r.read()

def main():
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
                procs.append(subprocess.Popen(cmd,cwd='/opt/ComfyUI',stdout=log,stderr=subprocess.STDOUT))
            base='http://127.0.0.1:8888/proxy/8188/'
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if any(p.poll() is not None for p in procs): raise RuntimeError('Service exited during startup')
                try:
                    status, body=get(base+'object_info',token)
                    info=json.loads(body)
                    assert 'UnetLoaderGGUF' in info
                    break
                except (OSError,ValueError,AssertionError): time.sleep(2)
            else: raise RuntimeError('CPU ComfyUI/proxy startup timed out')
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
