"""LRU cache."""
import collections
from typing import Any, List, Tuple, Callable
import numpy as np


class NumpyLRUCache:
    """Dict based LRU cache for numpy arrays."""

    def __init__(self, size):
        """Crate cache."""
        self.size = size
        self.lru_cache = collections.OrderedDict()
        self.common_dim = None

    def _get(self, key: Any, default=None) -> np.ndarray:
        try:
            value = self.lru_cache.pop(key)
            self.lru_cache[key] = value
            return value
        except KeyError:
            return default

    def _put(self, key: Any, value: np.ndarray):
        try:
            self.lru_cache.pop(key)
        except KeyError:
            if len(self.lru_cache) >= self.size:
                self.lru_cache.popitem(last=False)
            self.lru_cache[key] = value

    def cache_compute(
        self, keys: List[Any], function: Callable
    ) -> Tuple[np.ndarray, List[Any]]:
        """Get batch from cache and compute missing.

        Args:
            keys: List of keys

        Returns:
            np.ndarray or None: Found entries concatenated over 0 axis.
            list(_) or None: Keys that were not found.
        """
        if len(self.lru_cache) == 0:
            result = function(keys)
            self.put_batch(keys, result)
            return result
        else:
            result = np.zeros((len(keys), *self.common_dim))
            items = [self._get(key) for key in keys]
            not_found = [key for item, key in zip(items, keys) if item is None]
            if len(not_found) > 0:
                computed = function(not_found)
            computed_idx = 0
            for idx, item in enumerate(items):
                if item is None:
                    result_slc = tuple([idx] + [slice(None)] * len(self.common_dim))
                    computed_slc = tuple(
                        [computed_idx] + [slice(None)] * len(self.common_dim)
                    )
                    result[result_slc] = computed[computed_slc]
                    computed_idx += 1
                else:
                    result_slc = tuple([idx] + [slice(None)] * len(self.common_dim))
                    result[result_slc] = item
        return result

    def put_batch(self, keys: List[Any], values: np.ndarray):
        """Put batch to cache.

        Args:
            keys: List of keys
            values: Ndarray of shape (len(keys), *common_dim)
        """
        self._validate_shape(values)
        for key, item in zip(keys, values):
            self._put(key, item)

    def _validate_shape(self, value):
        if self.common_dim is None:
            self.common_dim = value.shape[1:]
        else:
            if self.common_dim != value.shape[1:]:
                raise ValueError(
                    f"""Cached arrays must have the same shape.
                    Shape expected: {self.common_dim}
                    Shape found: {value.shape[1:]}
                """
                )
