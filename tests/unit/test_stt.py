"""STT test."""
import numpy as np
from npc_engine.services.stt import SpeechToTextAPI
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, currentdir)
import mocks.zmq_mocks as zmq


class MockSTTModel(SpeechToTextAPI):
    def __init__(self) -> None:
        super().__init__(context=zmq.Context(), uri="inproc://test")
        self.i = 0

    def transcribe_frame(self, frame):
        if self.i == 0:
            o = "he"
        else:
            o = "lo"
        self.i += 1
        return o

    def transcribe(self, audio):
        return np.empty([0, 29])

    def postprocess(self, text):
        return "Hello"

    def decide_finished(self, text):
        return True

    def reset(self):
        pass

    def decode(self, logits):
        return "hello"


def test_stt_api():

    stt = MockSTTModel()

    o = stt.stt([1, 1, 1, 1])
    assert o == "Hello"
