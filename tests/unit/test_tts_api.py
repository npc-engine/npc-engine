"""TTS test."""
import numpy as np
from npc_engine.models.tts import TextToSpeechAPI
import pytest


class MockTTSModel(TextToSpeechAPI):
    def __init__(self) -> None:
        super().__init__()

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
