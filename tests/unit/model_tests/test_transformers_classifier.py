"""Similarity test."""
import os
from npc_engine import services
import time
import pytest
import yaml


path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "models")

subdirs = [
    f.path
    for f in os.scandir(path)
    if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
]

configs = [
    yaml.safe_load(open(os.path.join(subdir, "config.yml"), "r")) for subdir in subdirs
]

model_paths = [
    subdir
    for config, subdir in zip(configs, subdirs)
    if "TransformerSemanticSimilarity" in config["model_type"]
]

print(model_paths)


@pytest.mark.skipif(
    len(model_paths) == 0, reason="Model missing",
)
def test_transformers_similarity():
    """Check custom testing"""
    try:
        semantic_tests = services.BaseService.create(model_paths[0], None)
    except FileNotFoundError:
        return
    start = time.time()
    test_result = semantic_tests.compare(
        "Can I have a beer", ["Can I have a beer", "Give me a beer"]
    )
    print("custom test time elapsed", time.time() - start)
    assert len(test_result) == 2
