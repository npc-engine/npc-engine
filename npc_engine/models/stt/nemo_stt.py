"""Module that implements Huggingface transformers semantic similarity."""
from typing import List
import numpy as np
from os import path
from tokenizers import Tokenizer
from ctc_decoder import best_path
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from loguru import logger

from npc_engine.models.stt.stt_base import SpeechToTextAPI


class NemoSTT(SpeechToTextAPI):
    """Text to speech pipeline based on Nemo toolkit.

    Uses:

        - ONNX export of EncDecCTCModel from Nemo toolkit.
        - ONNX export of EncDecCTCModel preprocessor with cfg.preprocessor.stft_conv = True. 
        - Punctuation distillbert model from Nemo toolkit. (requires tokenizer.json as well)

    References:  
        https://github.com/NVIDIA/NeMo
        https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5
        https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html

    TODO:  
        cfg.preprocessor.stft_conv is going to be deprecated so:  

        - figure out parameters for librosa stft and mel spec generation
        - Use librosa instead of exported preprocessor.  


    ctc.onnx spec:  

        - inputs:  
            `audio_signal` mel spectogram of shape `(batch_size, 64, mel_sequence)`
        - outputs:  
            `tokens` of shape `(batch_size, token_sequence, logits)`  

    preprocess.onnx spec:  

        - inputs:  
            `signal` mel spectogram of shape `(batch_size, sequence)`
            `length` mel spectogram of shape `(batch_size,)`
        - outputs:  
            `mel` of shape `(batch_size, 64, mel_sequence)`  

    punctuation.onnx spec:  

        - inputs:  
            `input_ids` mel spectogram of shape `(batch_size, sequence)`
            `attention_mask` mel spectogram of shape `(batch_size, sequence)`
        - outputs: 
            `punctuation` of shape `(batch_size, sequence, 4)`  
            `capitalization` of shape `(batch_size, sequence, 2)`  
    """

    def __init__(
        self,
        timestep_duration: int,
        model_path: str,
        frame_size: int = 1000,
        frame_overlap: float = 2.5,
        offset: int = 10,
        sample_rate: int = 16000,
        *args,
        **kwargs,
    ):
        """Create and load biencoder model for semantic similarity.

        Args:
            model_path: A path where model config and weights are
            metric: distance to compute semantic similarity
        """
        super().__init__(
            sample_rate=sample_rate,
            model_path=model_path,
            frame_size=frame_size,
            *args,
            **kwargs,
        )
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_BASIC
        provider = rt.get_available_providers()[0]
        logger.info(f"STT uses {provider} provider")
        self.asr_model = rt.InferenceSession(
            path.join(model_path, "ctc.onnx"),
            providers=[provider],
            sess_options=sess_options,
        )
        self.preprocessor = rt.InferenceSession(
            path.join(model_path, "preprocess.onnx"),
            providers=[provider],
            sess_options=sess_options,
        )
        self.punctuation = rt.InferenceSession(
            path.join(model_path, "punctuation.onnx"),
            providers=[provider],
            sess_options=sess_options,
        )
        self.tokenizer = Tokenizer.from_file(path.join(model_path, "tokenizer.json"))
        self.asr_vocab = " abcdefghijklmnopqrstuvwxyz'"
        self.asr_vocab = list(self.asr_vocab)
        self.asr_vocab.append("")
        self.frame_size = int((frame_size / 1000) * self.sample_rate)
        self.n_frame_overlap = int(frame_overlap * self.sample_rate)
        self.offset = offset
        self.timestep_duration = timestep_duration
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(
            shape=int(2 * self.n_frame_overlap + self.frame_size), dtype=np.float32
        )
        self.prev_char = ""

        self.punct_labels = "O,.?"
        self.capit_labels = "OU"

    def transcribe(self, audio: List[float]) -> str:
        """Transcribe audio usign this pipeline.

        Args:
            audio: ndarray of int16 of shape (samples,).

        Returns:
            Transcribed text from the audio.
        """
        logits = self._predict(np.asarray(audio, dtype=np.float32))
        utterance = best_path(logits, self.asr_vocab)
        return utterance

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        length = np.asarray([audio.squeeze().shape[0]], np.float32)
        signal = audio.reshape([1, -1]).astype(np.float32)
        audio_signal = self.preprocessor.run(
            None, {"signal": signal, "length": length}
        )[0]
        return self.asr_model.run(None, {"audio_signal": audio_signal})[0][0]

    def transcribe_frame(self, frame: np.ndarray) -> str:
        """Transcribe audio usign this pipeline.

        Args:
            frame: ndarray of int16 of shape (frame_size,).

        Returns:
            Transcribed text from the audio.
        """
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Frame size incorrect: expected {self.frame_size}, got {len(frame)}"
            )
        self.buffer[: -self.frame_size] = self.buffer[self.frame_size :]
        self.buffer[-self.frame_size :] = frame
        logits = self._predict(self.buffer)
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap : -self.n_timesteps_overlap], self.asr_vocab
        )
        return self._greedy_merge(decoded[: len(decoded) - self.offset])

    def reset(self) -> str:
        """Reset frame_history and decoder's state."""
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ""

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ""
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i], axis=-1)]
        return s

    def _greedy_merge(self, s):
        s_merged = ""

        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != "_":
                    s_merged += self.prev_char
        return s_merged

    def postprocess(self, text: str) -> str:
        """Add punctuation and capitalization.

        Args:
            text: audio transcription.

        Returns:
            Postprocessed text transcribtion.
        """
        enc = self.tokenizer.encode(text)

        punct, capit = self.punctuation.run(
            None,
            {
                "input_ids": np.asarray(enc.ids, np.int64).reshape([1, -1]),
                "attention_mask": np.asarray(enc.attention_mask, np.int64).reshape(
                    [1, -1]
                ),
            },
        )
        punct = punct.argmax(2)
        capit = capit.argmax(2)
        punctuated_capitalized = self._apply_punct_capit_predictions(
            text, punct[0][1:-1].tolist(), capit[0][1:-1].tolist()
        )
        return punctuated_capitalized

    def _apply_punct_capit_predictions(
        self, query: str, punct_preds, capit_preds
    ) -> str:
        """Restores punctuation and capitalization in ``query``.

        Args:
            query: a string without punctuation and capitalization
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            a query with restored punctuation and capitalization
        """
        query = query.strip().split()

        query_with_punct_and_capit = ""
        skip = 0
        for j, word in enumerate(query):
            word_enc = self.tokenizer.encode(word, add_special_tokens=False).ids
            skip += len(word_enc) - 1
            punct_label = self.punct_labels[punct_preds[j + skip]]
            capit_label = self.capit_labels[capit_preds[j + skip]]

            if capit_label != "O":
                word = word.capitalize()
            query_with_punct_and_capit += word
            if punct_label != "O":
                query_with_punct_and_capit += punct_label
            query_with_punct_and_capit += " "
        return query_with_punct_and_capit[:-1]
