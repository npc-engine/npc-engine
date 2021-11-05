# -*- coding: utf-8 -*-
"""Text to speech specific model implementations.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.tts import TextToSpeech
        model = TextToSpeech.load("path/to/model_dir")
        model.run(speaker_id=0, text="Hello, world!")
    ```
"""

from .tts_base import TextToSpeech  # noqa: F401

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
