from typing import List
import numpy as np
import onnxruntime as rt
import os
from .text import text_to_sequence, cmudict, _clean_text, get_arpabet, acronyms
import re
import random


def get_text(text: str):
    text = _clean_text(text, ['english_cleaners'])
    words = re.findall(r'\S*\{.*?\}\S*|\S+', text)
    text = ' '.join([get_arpabet(word, acronyms.cmudict)
                        if random.random() < 0.5 else word
                        for word in words])
    text_norm = np.asarray(text_to_sequence(text), dtype=np.int64).reshape([1, -1])
    return text_norm

class TacotronSpeechSynthesizer:
    def __init__(self, tacotron_path):
        self.frames = 400
        self.sigma = 0.5
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.tts_net = rt.InferenceSession(
            os.path.join(tacotron_path, "flowtron_waveglow.onnx"),
            providers=rt.get_available_providers()[1:],
            sess_options=sess_options,
        )
        self.speaker_voices = {}

    def create_voice(self, speaker_id: str, voice_id: int):
        self.speaker_voices[speaker_id] = voice_id

    def delete_voice(self, speaker_id: str):
        del self.speaker_voices[speaker_id]

    def tts(self, speaker_id: str, line: str) -> np.ndarray:
        seq = get_text(line)
        rV = np.random.randn(1, 80, self.frames) * self.sigma
        inps = {
            "residual": rV.astype(np.float32),
            "text": seq,
            "speaker_vecs": np.asarray([[self.speaker_voices[speaker_id]]]).astype(np.int64)
        }
        audio = self.tts_net.run(None, inps)
        audio = audio[0].reshape(-1)
        audio = audio / np.abs(audio).max()
        return audio
