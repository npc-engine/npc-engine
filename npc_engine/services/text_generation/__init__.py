# -*- coding: utf-8 -*-
"""Text generation API services.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.services.text_generation import TextGenerationAPI
        model = TextGenerationAPI.load("path/to/model_dir")
        model.generate_reply(context, temperature=0.8, topk=None,)
    ```
"""
from .text_generation_base import TextGenerationAPI  # noqa: F401
from .hf_text_generation import HfChatbot  # noqa: F401
from .bart import BartChatbot  # noqa: F401
