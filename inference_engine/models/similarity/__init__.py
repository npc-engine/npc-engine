# -*- coding: utf-8 -*-
"""Similarity model implementations.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.tts import Similarity
        model = Similarity.load("path/to/model_dir")
        model.compare("hello", ["Hello, world!"])
    ```
"""

from .similarity_base import SimilarityAPI  # noqa: F401

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
