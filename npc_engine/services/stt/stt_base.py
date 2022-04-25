"""Module that implements speech to text model API."""
from typing import Dict, List


from abc import abstractmethod
from npc_engine.services.base_service import BaseService
import numpy as np
import sounddevice as sd
from queue import Queue
import webrtcvad
import re

# TODO: Add support for streaming audio from other processes


class SpeechToTextAPI(BaseService):
    """Abstract base class for speech to text models."""

    API_METHODS: List[str] = [
        "listen",
        "stt",
        "get_devices",
        "select_device",
        "initialize_microphone_input",
    ]

    def __init__(
        self,
        min_speech_duration=100,
        max_silence_duration=1000,
        vad_mode=None,
        sample_rate=16000,
        vad_frame_ms=10,
        pad_size=1000,
        *args,
        **kwargs,
    ):
        """Initialize VAD part of the API."""
        super().__init__(*args, **kwargs)
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
        self.vad_frame_size = int((vad_frame_ms * sample_rate) / 1000)
        self.running = False
        self.microphone_initialized = False
        self.silence_buffer = np.empty([0])
        self.pad_size = pad_size

    @classmethod
    def get_api_name(cls) -> str:
        """Get the API name."""
        return "SpeechToTextAPI"

    def __del__(self):
        """Stop listening on destruction."""
        if self.microphone_initialized:
            self.stream.stop()

    def initialize_microphone_input(self):
        """Initialize microphone."""
        if self.microphone_initialized:
            return
        self.running = False
        self.microphone_initialized = True

        def callback(in_data, frame_count, time_info, status):
            if self.running:
                try:
                    self.listen_queue.put(in_data.reshape(-1), block=False)
                except Exception:
                    return
            else:
                if not self._vad_frame(
                    in_data.reshape(-1)
                ):  # Register only silence for buffer
                    self.silence_buffer = np.append(
                        self.silence_buffer, in_data.reshape(-1)
                    )
                    self.silence_buffer = self.silence_buffer[-self.pad_size :]

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.vad_frame_size,
            callback=callback,
        )
        self.stream.start()
        while self.silence_buffer.shape[0] < self.pad_size:
            pass

    def listen(self, context: str = None) -> str:  # pragma: no cover
        """Listen for speech input and return text from speech when done.

        Listens for speech, if speech is active for longer than self.frame_size in milliseconds
        then starts transcribing it. On each voice activity detection (VAD) pause
        uses context to decide if transcribed text is a finished response to a context.
        If it is, applies preprocessing and returns the result.
        If transcribed text is not a response to a context but VAD pause persists through max_silence_duration
        then returns the results anyway.

        Requires a microphone input to be initialized.

        Args:
            context: A last line of the dialogue used to decide when to stop listening.
                It allows our STT system to not wait for a VAD timeout (max_silence_duration in ms).

        Returns:
            Recognized text from the audio.
        """
        if not self.microphone_initialized:
            raise RuntimeError("Microphone not initialized.")
        self.listen_queue.queue.clear()
        context = re.sub(r"[^A-Za-z0-9 ]+", "", context).lower() if context else None
        text = self._transcribe_vad_pause(context)
        processed = self.postprocess(text)

        self.listen_queue.queue.clear()
        return processed

    def _transcribe_vad_pause(self, context) -> str:
        done = False

        total_pause_ms = 0
        total_speech_ms = 0

        speech_appeared = False
        tested_pause = False

        logits = None
        signal = self.silence_buffer
        self.running = True
        while not done:
            try:
                vad_frame = self.listen_queue.get(block=False)
            except Exception:
                continue
            is_speech = self._vad_frame(vad_frame)

            total_speech_ms, total_pause_ms = self._update_vad_stats(
                is_speech, total_speech_ms, total_pause_ms
            )

            if total_speech_ms > self.min_speech_duration:
                speech_appeared = True

            if total_pause_ms == 0:
                tested_pause = False

            signal = np.append(signal, vad_frame)
            if not speech_appeared and total_speech_ms < self.min_speech_duration:
                #  Keep only last minimum detectable speech duration + buffer
                signal = signal[-self._ms_to_samplenum(self.min_speech_duration * 2) :]
            if signal.shape[0] >= self._ms_to_samplenum(self.min_speech_duration):
                if speech_appeared and total_pause_ms > self.max_silence_duration:
                    wrapped_signal = self._wrap_signal(signal)
                    logits = self.transcribe(wrapped_signal)
                    text = self.decode(logits)
                    done = True
                    self.running = False
                    return text
                elif (
                    speech_appeared
                    and total_pause_ms > self.min_speech_duration
                    and not tested_pause
                ):
                    tested_pause = True
                    logits = self.transcribe(np.pad(signal, (0, 1000), "wrap"))
                    text = self.decode(logits)
                    done = self.decide_finished(context, text)
                    self.running = not done

        self.running = False
        return text

    def stt(self, audio: List[int]) -> str:
        """Transcribe speech.

        Args:
            audio: PMC data with bit depth 16.

        Returns:
            Recognized text from the audio.
        """
        logits = self.transcribe(audio)
        text = self.decode(logits)
        text = self.postprocess(text)
        return text

    def get_devices(self) -> Dict[int, str]:  # pragma: no cover
        """Get available audio devices."""
        return [device["name"] for device in sd.query_devices()]

    def select_device(self, device_id: int):  # pragma: no cover
        """Get available audio devices."""
        device_id = int(device_id)
        if device_id >= len(sd.query_devices()) or device_id < 0:
            raise ValueError(
                f"Bad device id, valid device ids in range [0;{len(sd.query_devices())})"
            )
        sd.default.device = device_id

    def _wrap_signal(self, signal: np.ndarray) -> np.ndarray:
        """Append silence buffer at both ends of the signal."""
        signal_with_silence = np.append(
            np.append(self.silence_buffer, signal), self.silence_buffer
        )
        return signal_with_silence

    def _vad_frame(self, frame):  # pragma: no cover
        """Detect voice activity in a frame."""
        is_speech = self.vad.is_speech(
            np.frombuffer((frame * 32767).astype(np.int16).tobytes(), np.int8),
            sample_rate=16000,
        )
        return is_speech

    def _ms_to_samplenum(self, time):
        return int(time * self.sample_rate / 1000)

    def _samples_to_ms(self, samples):
        return samples * 1000 / self.sample_rate

    def _update_vad_stats(self, is_speech, total_speech_ms, total_pause_ms):
        """Increase total_speech_ms if is_speech is true and appears for min_speech_duration frames consequently."""
        if is_speech:
            total_speech_ms += self.vad_frame_ms
            total_pause_ms = 0
        else:
            total_pause_ms += self.vad_frame_ms
            total_speech_ms = 0
        return total_speech_ms, total_pause_ms

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> np.ndarray:
        """Abstract method for audio transcription.

        Should be implemented by the specific model.

        Args:
            audio: ndarray of int16 of shape (samples,).

        Returns:
            Transcribed logits from the audio.
        """
        return None

    @abstractmethod
    def decode(self, logits: np.ndarray) -> str:
        """Decode logits into text.

        Args:
            logits: ndarray of float32 of shape (timesteps, vocab_size).

        Returns:
            Decoded string.
        """
        return None

    @abstractmethod
    def decide_finished(self, context: str, text: str) -> bool:
        """Abstract method for deciding if audio transcription should be finished.

        Should be implemented by the specific model.

        Args:
            context: Text context of the speech recognized
                (e.g. a question to which speech recognized is a reply to).
            text: Recognized speech so far.

        Returns:
            Decision to stop recognition and finalize results.
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
