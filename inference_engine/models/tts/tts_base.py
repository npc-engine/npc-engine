from typing import Iterable, List

from abc import abstractmethod
from inference_engine.models.base_model import Model
import numpy as np


class TextToSpeech(Model):
    """Abstract base class for text-to-speech models.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.initialized = True

    @abstractmethod
    def run(self, speaker_id: str, text: str, n_chunks: int) -> Iterable[np.ndarray]:
        """Creates a generator for iterative generation of speech.

        Args:
            speaker_id: Id of the speaker.
            text: Text to generate speech from.
            n_chunks: Number of chunks to split generation into.

        Returns:
            Generator that yields next chunk of speech in the form of f32 ndarray.
        """
        return None

    @abstractmethod
    def get_speaker_ids(self) -> List[str]:
        """Gets list of available speaker ids.

        Returns:
            The return value. True for success, False otherwise.
        """
        return None
