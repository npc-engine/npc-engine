from typing import List
import numpy as np
import onnxruntime as rt
import os
from .text import text_to_sequence
import logging


def prepare_input_sequence(texts, cpu_run=False):

    d = []
    for i, text in enumerate(texts):
        d.append(np.asarray(
            text_to_sequence(text, ['english_cleaners'])[:], dtype=np.int64))
    input_length = np.asarray([len(x) for x in d], dtype=np.int64)
    return d, input_length


class TacotronSpeechSynthesizer:
    def __init__(self, tacotron_path):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.tts_net = rt.InferenceSession(
            os.path.join(tacotron_path, "tactron_full.onnx"),
            providers=[rt.get_available_providers()[1]],
            sess_options=sess_options,
        )

    def create_voice(self, speaker_id: int, traits: List[str] = None):
        pass

    def tts(self, speaker_id: int, line: str) -> np.ndarray:
        del speaker_id

        seq, lengths = prepare_input_sequence([line])
        inps = {"sequences": seq, "sequence_lengths": lengths}
        audio = self.tts_net.run(None, inps)
        return audio[0]
