# -*- coding: utf-8 -*-
"""Text to speech specific model implementations.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.tts import TextToSpeechAPI
        model = TextToSpeechAPI.load("path/to/model_dir")
        model.run(speaker_id=0, text="Hello, world!")
    ```
"""

from .tts_base import TextToSpeechAPI  # noqa: F401
from .flowtron import FlowtronTTS  # noqa: F401
