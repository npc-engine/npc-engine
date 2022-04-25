"""Module that implements sequence classification API."""
from typing import List

from abc import abstractmethod
from npc_engine.services.base_service import BaseService
from npc_engine.services.utils.lru_cache import NumpyLRUCache
import numpy as np


class SequenceClassifierAPI(BaseService):
    """Abstract base class for text classification models."""

    API_METHODS: List[str] = ["classify"]

    def __init__(self, cache_size=0, *args, **kwargs) -> None:
        """Empty initialization method for API to be similar to other model base classes."""
        super().__init__(*args, **kwargs)
        self.initialized = True
        self.cache = NumpyLRUCache(cache_size)

    @classmethod
    def get_api_name(cls) -> str:
        """Get the API name."""
        return "SequenceClassifierAPI"

    def classify(self, texts: List[str]) -> List[List[float]]:
        """Classify a list of texts.

        Args:
            texts: A list of texts to classify.

        Returns:
            List of scores for each text.
        """
        texts = [row if isinstance(row, str) else tuple(row) for row in texts]
        scores = self.cache.cache_compute(
            texts, lambda values: self.compute_scores_batch(values)
        )
        return scores.tolist()

    @abstractmethod
    def compute_scores_batch(self, texts: List[str]) -> np.ndarray:
        """Compute scores for a list of texts.

        Args:
            texts: A list of texts to compute scores for.

        Returns:
            List of scores for each text.
        """
        pass
