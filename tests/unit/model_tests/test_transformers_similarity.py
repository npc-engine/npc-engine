"""Similarity test."""
import os
from npc_engine import models
import time
import pytest
import yaml


path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "npc_engine", "resources", "models"
)

subdirs = [
    f.path
    for f in os.scandir(path)
    if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
]

configs = [
    yaml.load(open(os.path.join(subdir, "config.yml"), "r")) for subdir in subdirs
]

nemo_stt_paths = [
    subdir
    for config, subdir in zip(configs, subdirs)
    if "TransformerSemanticSimilarity" in config["model_type"]
]


@pytest.mark.skipif(
    len(nemo_stt_paths) == 0, reason="Model missing",
)
def test_transformers_similarity():
    """Check custom testing"""
    try:
        semantic_tests = models.Model.load(nemo_stt_paths[0])
    except FileNotFoundError:
        return
    start = time.time()
    test_result = semantic_tests.compare(
        "Can I have a beer", ["Can I have a beer", "Give me a beer"]
    )
    print("custom test time elapsed", time.time() - start)
    assert len(test_result) == 2
