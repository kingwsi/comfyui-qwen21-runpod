"""Template-based image generation UI, served inside ComfyUI."""
from .api import register
from server import PromptServer
import folder_paths

register(PromptServer.instance.routes, folder_paths)
NODE_CLASS_MAPPINGS = {}
