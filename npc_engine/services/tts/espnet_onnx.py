"""[espnet_onnx](https://github.com/Masao-Someki/espnet_onnx) text to speech inference implementation."""
from typing import Iterator, List
import numpy as np
from espnet_onnx import Text2Speech
from npc_engine.services.tts.tts_base import TextToSpeechAPI
import logging


class ESPNetTTS(TextToSpeechAPI):
    """Service implementation for the espnet_onnx text to speech models."""

    def __init__(self, model_path: str, speaker_num: int, *args, **kwargs):
        """Create and load espnet_onnx Text2Speech model.

        Args:
            model_path: Path to the model directory.
            speaker_num: Number of speakers model supports.
        """
        super().__init__(*args, **kwargs)
        provider = self.get_providers()
        self.t2s = Text2Speech(model_path, providers=provider)
        logging.info("ESPNetTTS using providers {}".format(provider))
        if not self.t2s.tts_model.use_sids:
            self.speaker_ids = range(speaker_num)
        else:
            self.speaker_ids = list()

    def get_speaker_ids(self) -> List[str]:
        """Return available ids of different speakers."""
        return self.speaker_ids

    def run(self, speaker_id: str, text: str, n_chunks: int) -> Iterator[np.ndarray]:
        """Create a generator for iterative generation of speech.

        Args:
            speaker_id: Id of the speaker.
            text: Text to generate speech from.
            n_chunks: Number of chunks to split generation into.

        Returns:
            Generator that yields next chunk of speech in the form of f32 ndarray.
        """
        try:
            speaker_id = int(speaker_id)
        except ValueError:
            raise ValueError("Speaker id in espnet models must be an integer")
        if speaker_id not in self.speaker_ids:
            raise ValueError("Speaker id {} not supported".format(speaker_id))
        return iter([self.t2s(text, sids=np.asarray([int(speaker_id)]))["wav"]])
