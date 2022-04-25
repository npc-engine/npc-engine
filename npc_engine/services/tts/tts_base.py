"""Module that implements text to speech model API."""
from typing import Iterable, List

from abc import abstractmethod
from npc_engine.services.base_service import BaseService
import numpy as np
import re


class TextToSpeechAPI(BaseService):
    """Abstract base class for text-to-speech models."""

    #: Methods that are going to be exposed as services.
    API_METHODS: List[str] = ["tts_start", "tts_get_results", "get_speaker_ids"]

    def __init__(self, *args, **kwargs) -> None:
        """Empty initialization method for API to be similar to other model base classes."""
        self.generator = None
        super().__init__(*args, **kwargs)
        self.initialized = True

    @classmethod
    def get_api_name(cls) -> str:
        """Get the API name."""
        return "TextToSpeechAPI"

    def tts_start(self, speaker_id: str, text: str, n_chunks: int) -> None:
        """Initiate iterative generation of speech.

        Args:
            speaker_id: Id of the speaker.
            text: Text to generate speech from.
            n_chunks: Number of chunks to split generation into.

        """
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        self.generator = self._chain_run(speaker_id, sentences, n_chunks)

    def _chain_run(self, speaker_id, sentences, n_chunks) -> Iterable[np.ndarray]:
        """Chain the run method to be used in the generator."""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence != "":
                yield from self.run(speaker_id, sentence, n_chunks)

    def tts_get_results(self) -> Iterable[np.ndarray]:
        """Retrieve the next chunk of generated speech.

        Returns:
            Next chunk of speech in the form of f32 ndarray.
        """
        if self.generator is not None:
            return next(self.generator).tolist()
        else:
            raise ValueError(
                "Speech generation was not started. Use tts_start to start it"
            )

    @abstractmethod
    def run(self, speaker_id: str, text: str, n_chunks: int) -> Iterable[np.ndarray]:
        """Create a generator for iterative generation of speech.

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
        """Get list of available speaker ids.

        Returns:
            The return value. True for success, False otherwise.
        """
        return None
