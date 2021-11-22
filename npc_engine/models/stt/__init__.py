# -*- coding: utf-8 -*-
"""Speech to text API.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.stt import SpeechToTextAPI
        model = SpeechToTextAPI.load("path/to/model_dir")
        text = model.listen()  # Say something
    ```
"""

from .stt_base import SpeechToTextAPI  # noqa: F401

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
