"""Module that implements semantic similarity model API."""
from typing import Dict, List

from abc import abstractmethod
from npc_engine.models.base_model import Model
from npc_engine.models.utils.lru_cache import NumpyLRUCache
import numpy as np


class SimilarityAPI(Model):
    """Abstract base class for text classification models."""

    API_METHODS = ["compare", "cache"]

    def __init__(self, cache_size=0, *args, **kwargs) -> None:
        """Empty initialization method for API to be similar to other model base classes."""
        super().__init__()
        self.initialized = True
        self.lru_cache = NumpyLRUCache(cache_size)

    def compare(self, query: str, context: List[str]) -> Dict[str, float]:
        """Compare a query to the context.

        Args:
            query: A sentence to compare.
            context: A list of sentences to compare to. This will be cached if caching is enabled

        Returns:
            List of similarities
        """
        embedding_a = self.compute_embedding(query)
        embedding_b = self.lru_cache.cache_compute(
            context, lambda values: self.compute_embedding_batch(values)
        )
        similarities = self.metric(embedding_a, embedding_b)
        return similarities.tolist()

    def cache(self, context: List[str]):
        """Compare a query to the context.

        Args:
            query: A sentence to compare.
            context: A list of sentences to compare to. This will be cached if caching is enabled

        Returns:
            List of similarities
        """
        self.lru_cache.cache_compute(
            context, lambda values: self.compute_embedding_batch(values)
        )

    @abstractmethod
    def compute_embedding_batch(self, lines: List[str]) -> np.ndarray:
        """Compute line embeddings in batch.

        Args:
            lines: List of sentences to embed

        Returns:
            Embedding batch of shape (batch_size, embedding_size)
        """
        return None

    @abstractmethod
    def compute_embedding(self, line: str) -> np.ndarray:
        """Compute sentence embedding.

        Args:
            line: Sentence to embed

        Returns:
            Embedding of shape (1, embedding_size)
        """
        return None

    @abstractmethod
    def metric(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> np.ndarray:
        """Compute distance between two embeddings.

        Embeddings are of broadcastable shapes. (1 or batch_size)
        Args:
            embedding_a: Embedding of shape (1 or batch_size, embedding_size)
            embedding_b: Embedding of shape (1 or batch_size, embedding_size)

        Returns:
            Vector of distances (batch_size or 1,)
        """
        return None
