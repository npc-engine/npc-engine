"""Similarity test."""
import os
from npc_engine import services
import time
import pytest
import yaml
import inspect
import sys

from npc_engine.services.utils.config import get_type_from_dict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import mocks.zmq_mocks as zmq

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
    if "HfClassifier" in get_type_from_dict(config)
]


@pytest.mark.skipif(
    len(model_paths) == 0,
    reason="Model missing",
)
def test_transformers_classification():
    """Check custom testing"""
    try:
        semantic_tests = services.BaseService.create(
            zmq.Context(), model_paths[0], "inproc://test", service_id="test"
        )
    except FileNotFoundError:
        return
    start = time.time()
    test_result = semantic_tests.classify(["hello", ("world", "world")])
    print("custom test time elapsed", time.time() - start)
    assert len(test_result) == 2
