"""Frame-accurate multilingual opening renderer."""

from .errors import LangcutError
from .manifest import load_manifest
from .render import render

__all__ = ["LangcutError", "load_manifest", "render"]
__version__ = "1.0.0"
