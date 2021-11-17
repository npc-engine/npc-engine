"""Module that implements speech to text model API."""
from typing import Dict, List

import os
from abc import abstractmethod
from npc_engine.models.base_model import Model
import numpy as np
import sounddevice as sd
from queue import Queue
import webrtcvad
from loguru import logger
from tokenizers import Tokenizer
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
import re


class SpeechToTextAPI(Model):
    """Abstract base class for speech to text models."""

    API_METHODS: List[str] = ["listen", "stt", "get_devices", "select_device"]

    def __init__(
        self,
        model_path,
        min_speech_duration,
        max_silence_duration,
        vad_mode=None,
        sample_rate=16000,
        vad_frame_ms=10,
        frame_size=1000,
        transcribe_realtime=True,
        *args,
        **kwargs,
    ):
        """Initialize VAD part of the API."""
        super().__init__()
        self.initialized = True
        sd.default.samplerate = sample_rate

        self.max_silence_duration = max_silence_duration
        self.min_speech_duration = min_speech_duration
        self.vad = webrtcvad.Vad()
        if vad_mode is not None:
            self.vad.set_mode(vad_mode)
        self.sample_rate = sample_rate
        self.listen_queue = Queue(10)
        self.vad_frame_ms = vad_frame_ms
        self.frame_size_sampling = frame_size
        self.transcribe_realtime = transcribe_realtime
        self.vad_frame_size = int((vad_frame_ms * sample_rate) / 1000)

        if transcribe_realtime:
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
            self.sentence_model = rt.InferenceSession(
                os.path.join(model_path, "sentence_prediction.onnx"),
                providers=[rt.get_available_providers()[0]],
                sess_options=sess_options,
            )
            self.sentence_tokenizer = Tokenizer.from_file(
                os.path.join(model_path, "sentence_tokenizer.json")
            )
            logger.info(
                f"Sentence classifier uses {rt.get_available_providers()[0]} provider"
            )

    def listen(self, context: str) -> str:
        """Listen for speech input and return text from speech when done.

        Listens for speech, if speech is active for longer than self.frame_size in milliseconds
        then starts transcribing it. On each voice activity detection (VAD) pause 
        uses context to decide if transcribed text is a finished response to a context. 
        If it is applies preprocessing and returns the result.
        If transcribed text is not a response to a context but VAD pause persists through max_silence_duration
        then returns the results anyway.

        Args:
            context: A last line of the dialogue used to decide when to stop listening.
                It allows our STT system to not wait for a VAD timeout (max_silence_duration in ms).

        Returns:
            Recognized text from the audio.
        """
        self.listen_queue.queue.clear()
        self.reset()
        context = re.sub(r"[^A-Za-z0-9 ]+", "", context).lower()

        def callback(in_data, frame_count, time_info, status):
            # signal = sps.resample(signal, int((self.frame_size / 1000) * self.sample_rate)
            self.listen_queue.put(in_data.reshape(-1))

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=int((self.frame_size_sampling / 1000) * self.sample_rate),
            callback=callback,
        ):
            done = False
            total_pause_ms = 0
            transcribing = False
            text = ""
            while not done:
                frame = self.listen_queue.get(
                    block=True, timeout=(self.frame_size * 2) / 1000
                )
                speech_present, pause_length = self._vad_frame(frame)
                if speech_present:
                    total_pause_ms = 0
                else:
                    total_pause_ms += pause_length

                if not transcribing:
                    transcribing = speech_present

                if transcribing:
                    text += self.transcribe_frame(frame)
                    if total_pause_ms > self.max_silence_duration:
                        done = True
                    if total_pause_ms > self.min_speech_duration and not done:
                        decision = self._decide_sentence_finished(context, text)
                        done = bool(decision)
        processed = self.postprocess(text)

        self.listen_queue.queue.clear()
        self.reset()
        return processed

    def stt(self, audio: List[int]) -> str:
        """Transcribe speech.

        Args:
            audio: PMC data with bit depth 16.

        Returns:
            Recognized text from the audio.
        """
        text = self.transcribe(audio)
        text = self.postprocess(text)
        return text

    def get_devices(self) -> Dict[int, str]:
        """Get available audio devices."""
        return [device["name"] for device in sd.query_devices()]

    def select_device(self, device_id: int):
        """Get available audio devices."""
        device_id = int(device_id)
        if device_id >= len(sd.query_devices()) or device_id < 0:
            raise ValueError(
                f"Bad device id, valid device ids in range [0;{len(sd.query_devices())})"
            )
        sd.default.device = device_id

    def _decide_sentence_finished(self, context, text):
        tokenized = self.sentence_tokenizer.encode(context, text)
        ids = np.asarray(tokenized.ids).reshape([1, -1]).astype(np.int64)
        type_ids = np.asarray(tokenized.type_ids).reshape([1, -1]).astype(np.int64)
        attention_mask = np.ones_like(ids)
        input_dict = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }
        logits = self.sentence_model.run(None, input_dict)[0]
        return logits.argmax(-1)[0]

    def _vad_frame(self, frame):
        vad_frames = frame[
            : frame.shape[0] // self.vad_frame_size * self.vad_frame_size
        ].reshape([-1, self.vad_frame_size])
        speech_present = False
        speech_total = 0
        pause_total = 0
        for vad_frame in vad_frames:
            is_speech = self.vad.is_speech(
                np.frombuffer((vad_frame * 32767).astype(np.int16).tobytes(), np.int8),
                sample_rate=16000,
            )
            if is_speech:
                speech_total += self.vad_frame_ms
            else:
                speech_total = 0

            if speech_total >= self.min_speech_duration:
                pause_total = 0
                speech_present = True
            else:
                pause_total += self.vad_frame_ms

        return speech_present, pause_total

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Abstract method for audio transcription.

        Should be implemented by the specific model.

        Args:
            audio: ndarray of int16 of shape (samples,).

        Returns:
            Transcribed text from the audio.
        """
        return None

    @abstractmethod
    def transcribe_frame(self, frame: np.ndarray) -> str:
        """Abstract method for audio transcription iteratively.

        Should be implemented by the specific model.

        Args:
            frame: ndarray of int16 of shape (frame_size,).

        Returns:
            Transcribed text from the audio.
        """
        return None

    @abstractmethod
    def reset(self):
        """Abstract method for resetting iterative audio transcription state.

        Should be implemented by the specific model.
        """
        return None

    @abstractmethod
    def postprocess(self, text: str) -> str:
        """Abstract method for audio transcription postprocessing.

        Should be implemented by the specific model.

        Args:
            text: audio transcription.

        Returns:
            Postprocessed text transcribtion.
        """
        return None
