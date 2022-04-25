# -*- coding: utf-8 -*-
"""Chatbot model implementations.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.tts import Similarity
        model = Similarity.load("path/to/model_dir")
        model.generate_repy(context, temperature=0.8, topk=None,)
    ```
"""
from .chatbot_base import ChatbotAPI  # noqa: F401

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
