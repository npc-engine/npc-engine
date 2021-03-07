import numpy as np
import logging
from inference_engine.models.tts.flowtron import FlowtronTTS


class TTS:
    def __init__(self, model_path):
        self.tts_net = FlowtronTTS(model_path)

    def tts(self, voice_id: int, line: str) -> np.ndarray:
        try:
            voice_id = np.asarray([[voice_id]]).astype(np.int64)
            audio = self.tts_net.run(voice_id, line)
        except Exception as e:
            logging.error("TTS excepiton happened", e)
            audio = [np.empty(shape=[0], dtype=np.float32)]
        return audio
