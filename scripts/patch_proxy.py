"""Apply the reviewed encoded-slash fix only to the pinned proxy package."""
from importlib.metadata import version
from pathlib import Path
import jupyter_server_proxy
assert version('jupyter-server-proxy') == '4.6.0'
path = Path(jupyter_server_proxy.__file__).parent / 'handlers.py'
text = path.read_text()
anchor = '''        client_path = quote(client_path, safe=":/?#[]@!$&'()*+,;=-._~")\n'''
assert text.count(anchor) == 1, 'Unexpected upstream proxy source; refusing patch'
addition = '''
        # Tornado decodes captured groups; keep raw encoded separators for
        # ComfyUI userdata paths. Authentication and host checks are unchanged.
        raw_prefix = url_path_join(self.settings.get("base_url", "/"), "proxy", str(port)) + "/"
        if (not self.absolute_url and host == "localhost"
                and self.request.path.startswith(raw_prefix)):
            client_path = "/" + self.request.path[len(raw_prefix):]
'''
path.write_text(text.replace(anchor, anchor + addition))
