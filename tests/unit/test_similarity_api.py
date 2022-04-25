"""Similarity test."""
import numpy as np
from npc_engine.models.similarity import SimilarityAPI

num_calls = 0


class MockSimilarityModel(SimilarityAPI):
    def __init__(self) -> None:
        super().__init__(10)

    def compute_embedding(self, line):
        return np.asarray([123]).reshape(1, 1)

    def compute_embedding_batch(self, lines):
        global num_calls
        num_calls += 1
        return np.asarray([num_calls]).reshape([1, 1])

    def metric(self, embedding_a, embedding_b):
        assert embedding_a == np.asarray([123]).reshape(1, 1)
        assert embedding_b == np.asarray([1]).reshape([1, 1])
        return np.asarray([1.0])


def test_similarity_api():
    """Check custom testing"""

    semantic_tests = MockSimilarityModel()

    test_result = semantic_tests.cache(["Give me a beer"])
    test_result = semantic_tests.compare("Can I have a beer", ["Give me a beer"])
