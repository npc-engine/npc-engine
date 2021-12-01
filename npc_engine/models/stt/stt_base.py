"""Module that implements speech to text model API."""
from typing import Dict, List


from abc import abstractmethod
from npc_engine.models.base_model import Model
import numpy as np
import sounddevice as sd
from queue import Queue
import webrtcvad
import re


class SpeechToTextAPI(Model):  # pragma: no cover
    """Abstract base class for speech to text models."""

    API_METHODS: List[str] = ["listen", "stt", "get_devices", "select_device"]

    def __init__(
        self,
        min_speech_duration=100,
        max_silence_duration=1000,
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

    def listen(self, context: str = None) -> str:  # pragma: no cover
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
        context = re.sub(r"[^A-Za-z0-9 ]+", "", context).lower() if context else None
        if self.transcribe_realtime:
            text = self._transcribe_realtime(context)
        else:
            text = self._transcribe_vad_pause()
        processed = self.postprocess(text)

        self.listen_queue.queue.clear()
        self.reset()
        return processed

    def _transcribe_realtime(self, context: str) -> str:
        def callback(in_data, frame_count, time_info, status):
            # signal = sps.resample(signal, int((self.frame_size / 1000) * self.sample_rate)
            self.listen_queue.put(in_data.reshape(-1))

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.vad_frame_size,
            callback=callback,
        ):
            done = False

            total_pause_ms = 0
            total_speech_ms = 0

            speech_appeared = False

            asr_frame = np.empty([0], np.float32)
            logits = None
            while not done:
                vad_frame = self.listen_queue.get(
                    block=True, timeout=self._samples_to_ms(self.frame_size * 2)
                )
                is_speech = self._vad_frame(vad_frame)

                total_speech_ms, total_pause_ms = self._update_vad_stats(
                    is_speech, total_speech_ms, total_pause_ms
                )
                asr_frame = np.append(asr_frame, vad_frame)
                if not speech_appeared and total_speech_ms < self.min_speech_duration:
                    #  Keep only last minimum detectable speech duration
                    asr_frame = asr_frame[
                        -self._ms_to_samplenum(self.min_speech_duration) :
                    ]
                else:
                    speech_appeared = True

                if speech_appeared and total_pause_ms > self.max_silence_duration:
                    done = True
                    asr_frame = np.pad(
                        asr_frame,
                        (
                            0,
                            self._ms_to_samplenum(self.frame_size_sampling)
                            - asr_frame.shape[0],
                        ),
                        "constant",
                    )

                if speech_appeared and asr_frame.shape[0] >= self._ms_to_samplenum(
                    self.frame_size_sampling
                ):
                    print(f"\r speech_appeared {speech_appeared}")
                    if logits is None:
                        logits = self.transcribe(asr_frame)
                    else:
                        logits = np.concatenate((logits, self.transcribe(asr_frame)))
                    text = self.decode(logits)
                    if not done and total_pause_ms > 0:
                        done = self.decide_finished(context, text)
                    asr_frame = np.empty([0], np.float32)
        return text

    def _transcribe_vad_pause(self) -> str:
        def callback(in_data, frame_count, time_info, status):
            # signal = sps.resample(signal, int((self.frame_size / 1000) * self.sample_rate)
            self.listen_queue.put(in_data.reshape(-1))

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.vad_frame_size,
            callback=callback,
        ):
            done = False

            total_pause_ms = 0
            total_speech_ms = 0

            speech_appeared = False

            logits = None
            signal = np.empty([0], np.float32)
            while not done:
                vad_frame = self.listen_queue.get(
                    block=True, timeout=self._samples_to_ms(self.frame_size * 2)
                )
                is_speech = self._vad_frame(vad_frame)

                total_speech_ms, total_pause_ms = self._update_vad_stats(
                    is_speech, total_speech_ms, total_pause_ms
                )

                if total_speech_ms > self.min_speech_duration:
                    speech_appeared = True

                print(f"\r speech_appeared {speech_appeared}")

                signal = np.append(signal, vad_frame)
                if not speech_appeared and total_speech_ms < self.min_speech_duration:
                    #  Keep only last minimum detectable speech duration
                    signal = signal[-self._ms_to_samplenum(self.min_speech_duration) :]
                else:
                    speech_appeared = True

                if speech_appeared and total_pause_ms > self.max_silence_duration:
                    done = True
                    logits = self.transcribe(signal)
                    text = self.decode(logits)
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
    def transcribe_frame(self, frame: np.ndarray) -> np.ndarray:
        """Abstract method for audio transcription iteratively.

        Should be implemented by the specific model.

        Args:
            frame: ndarray of int16 of shape (frame_size,).

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
