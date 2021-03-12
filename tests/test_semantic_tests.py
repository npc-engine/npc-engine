"""
.. currentmodule:: test_dialog_script
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Dialog scripted test.
"""
import os
import simpleaudio as sa
from inference_engine.semantic_tests import SemanticTests
import time
import logging


def test_custom_test():
    """Check custom testing
    """
    semantic_tests = SemanticTests(
        os.path.join(
            os.path.dirname(__file__),
            "..\\inference_engine\\resources\\models\\roberta_semb",
        )
    )
    start = time.time()
    test_result = semantic_tests.test_custom(
        "Can I have a beer",
        ["Can I have a beer", "Give me a beer"],
        method=semantic_tests.OR_METHOD,
    )
    print("custom test time elapsed", time.time() - start)
    assert test_result == 1


def test_custom_():
    """Check predefined test
    """
    semantic_tests = SemanticTests(
        os.path.join(
            os.path.dirname(__file__),
            "..\\inference_engine\\resources\\models\\roberta_semb",
        )
    )
    semantic_tests.add_test(
        "beer", ["Can I have a beer", "Give me a beer"],
    )
    start = time.time()
    test_result = semantic_tests.test(
        "Can I have a beer", ["beer"], semantic_tests.OR_METHOD
    )
    print("predefined test time elapsed", time.time() - start)
    print(test_result)
    assert test_result["beer"] == 1
