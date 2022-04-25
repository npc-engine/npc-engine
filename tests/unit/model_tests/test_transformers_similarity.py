"""Similarity test."""
import os
from npc_engine import models
import time
import pytest

model_fp = os.path.join(
    os.path.dirname(__file__),
    "..\\..\\..\\npc_engine\\resources\\models\\roberta_semb",
)


@pytest.mark.skipif(
    not os.path.exists(model_fp), reason="Model missing",
)
def test_transformers_similarity():
    """Check custom testing"""
    try:
        semantic_tests = models.Model.load(model_fp)
    except FileNotFoundError:
        return
    start = time.time()
    test_result = semantic_tests.compare(
        "Can I have a beer", ["Can I have a beer", "Give me a beer"]
    )
    print("custom test time elapsed", time.time() - start)
    assert len(test_result) == 2
