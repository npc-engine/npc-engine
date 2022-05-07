"""TTS test."""
import numpy as np
from npc_engine.services.tts import TextToSpeechAPI

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
import mocks.zmq_mocks as zmq
import pytest


class MockTTSModel(TextToSpeechAPI):
    def __init__(self) -> None:
        super().__init__(context=zmq.Context(), service_id="test", uri="inproc://test")

    def run(self, speaker_id: str, text: str, n_chunks: int):
        return iter([np.asarray([123]).reshape(1, 1)])

    def get_speaker_ids(self):
        return ["1"]


def test_tts_api():

    tts = MockTTSModel()
    with pytest.raises(ValueError):
        test_result = tts.tts_get_results()

    tts.tts_start("0", "test", 10)
    test_result = tts.tts_get_results()
    assert np.asarray([123]).reshape(1, 1) == test_result
    assert ["1"] == tts.get_speaker_ids()
