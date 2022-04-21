# -*- coding: utf-8 -*-
"""Similarity model implementations.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.similarity import SimilarityAPI
        model = SimilarityAPI.load("path/to/model_dir")
        model.compare("hello", ["Hello, world!"])
    ```
"""

from .similarity_base import SimilarityAPI  # noqa: F401
from .similarity_transformers import TransformerSemanticSimilarity  # noqa: F401
