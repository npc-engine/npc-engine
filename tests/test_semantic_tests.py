"""
.. currentmodule:: test_dialog_script
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Dialog scripted test.
"""
import os
import simpleaudio as sa
from inference_engine import models
import time
import logging


def test_custom_test():
    """Check custom testing
    """
    semantic_tests = models.Model.load(
        os.path.join(
            os.path.dirname(__file__),
            "..\\inference_engine\\resources\\models\\roberta_semb",
        )
    )
    start = time.time()
    test_result = semantic_tests.compare(
        "Can I have a beer", ["Can I have a beer", "Give me a beer"]
    )
    print("custom test time elapsed", time.time() - start)
    assert len(test_result) == 2
