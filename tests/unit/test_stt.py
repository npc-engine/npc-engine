"""TTS test."""
import numpy as np
from npc_engine.models.stt import SpeechToTextAPI
import pytest


class MockSTTModel(SpeechToTextAPI):
    def __init__(self) -> None:
        super().__init__()
        self.i = 0

    def transcribe_frame(self, frame):
        if self.i == 0:
            o = "he"
        else:
            o = "lo"
        self.i += 1
        return o

    def transcribe(self, audio):
        return "hello"

    def postprocess(self, text):
        return "Hello"


def test_tts_api():
    """Check custom testing"""

    stt = MockSTTModel()

    o = stt.stt([1, 1, 1, 1])
    assert o == "Hello"
