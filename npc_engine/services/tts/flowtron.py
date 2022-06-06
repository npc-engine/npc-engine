"""Flowtron (https://github.com/NVIDIA/flowtron) text to speech inference implementation."""
from os import path
from typing import Iterator, List
import numpy as np
import onnxruntime
from npc_engine.services.tts.tts_base import TextToSpeechAPI
from npc_engine.text import (
    text_to_sequence,
    _clean_text,
)
import re
import logging


class FlowtronTTS(TextToSpeechAPI):
    """Implements Flowtron architecture inference.

    Paper:
    [arXiv:2005.05957](https://arxiv.org/abs/2005.05957)
    Code:
    https://github.com/NVIDIA/flowtron

    Onnx export script can be found in this fork https://github.com/npc-engine/flowtron.

    This model class requires four ONNX models `encoder.onnx`, `backward_flow.onnx`,
    `forward_flow.onnx` and `vocoder.onnx` where first three are layers from Flowtron
    architecture (`flow` corresponding to one direction pass of affine coupling layers)
    and `vocoder.onnx` is neural vocoder.

    For detailed specs refer to https://github.com/npc-engine/flowtron.
    """

    def __init__(
        self,
        model_path,
        max_frames=400,
        gate_threshold=0.5,
        sigma=0.8,
        smoothing_window=3,
        smoothing_weight=0.5,
        *args,
        **kwargs
    ):
        """Create and load Flowtron and vocoder models."""
        super().__init__(*args, **kwargs)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        provider = onnxruntime.get_available_providers()[0]
        logging.info("FlowtronTTS using provider {}".format(provider))
        self.max_frames = max_frames
        self.gate_threshold = gate_threshold
        self.sigma = sigma
        self.smoothing_window = smoothing_window
        self.smoothing_weight = smoothing_weight

        self.encoder = onnxruntime.InferenceSession(
            path.join(model_path, "encoder.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.backward_flow = onnxruntime.InferenceSession(
            path.join(model_path, "backward_flow.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.forward_flow = onnxruntime.InferenceSession(
            path.join(model_path, "forward_flow.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.vocoder = onnxruntime.InferenceSession(
            path.join(model_path, "vocoder.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.speaker_ids = [str(i) for i in range(127)]
        self.speaker_ids_map = {idx: i for i, idx in enumerate(self.speaker_ids)}

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
        text = self._get_text(text)
        speaker_id = np.asarray([[self.speaker_ids_map[speaker_id]]], dtype=np.int64)
        enc_outps_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            [text.shape[1], 1, 640], np.float32, "cpu", 0
        )

        io_binding = self.encoder.io_binding()
        io_binding.bind_ortvalue_output("text_emb", enc_outps_ortvalue)
        io_binding.bind_cpu_input("speaker_vecs", speaker_id)
        io_binding.bind_cpu_input("text", text.reshape([1, -1]))
        self.encoder.run_with_iobinding(io_binding)

        residual = np.random.normal(
            0, self.sigma, size=[self.max_frames, 1, 80]
        ).astype(np.float32)

        residual = self._run_backward_flow(residual, enc_outps_ortvalue)
        residual = self._run_forward_flow(
            residual, enc_outps_ortvalue, num_split=self.max_frames // n_chunks
        )
        last_audio = None
        for residual in residual:
            residual = np.transpose(residual, axes=(1, 2, 0))
            audio = self.vocoder.run(None, {"mels": residual})[0]
            # audio = np.where(
            #     (audio > (audio.mean() - audio.std()))
            #     | (audio < (audio.mean() + audio.std())),
            #     audio,
            #     audio.mean(),
            # )
            tmp = audio
            if last_audio is None:
                audio = audio[:, 1000:]
            if last_audio is not None:
                cumsum_vec = np.cumsum(
                    np.concatenate([last_audio, audio], axis=1), axis=1
                )
                ma_vec = (
                    cumsum_vec[:, self.smoothing_window :]
                    - cumsum_vec[:, : -self.smoothing_window]
                ) / self.smoothing_window
                audio = (1 - self.smoothing_weight) * audio[
                    :, self.smoothing_window :
                ] + self.smoothing_weight * ma_vec[:, last_audio.shape[1] :]
            last_audio = tmp
            audio = audio.reshape(-1)
            # audio = audio / np.abs(audio).max()
            yield audio

    def _get_text(self, text: str):
        text = _clean_text(text, ["flowtron_cleaners"])
        words = re.findall(r"\S*\{.*?\}\S*|\S+", text)
        text = " ".join(words)
        text_norm = np.asarray(text_to_sequence(text), dtype=np.int64).reshape([1, -1])
        return text_norm

    def _run_backward_flow(self, residual, enc_outps_ortvalue):

        residual_o, hidden_att, hidden_lstm = self._init_states(residual)

        hidden_att_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        hidden_att_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        residual_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            residual_o, "cpu", 0
        )

        residual_outp = [residual_ortvalue]

        for i in range(residual.shape[0] - 1, -1, -1):

            io_binding = self.backward_flow.io_binding()

            io_binding.bind_cpu_input("residual", residual[i])

            io_binding.bind_ortvalue_input("text", enc_outps_ortvalue)
            io_binding.bind_ortvalue_input("last_output", residual_outp[0])

            io_binding.bind_ortvalue_input("hidden_att", hidden_att_ortvalue)
            io_binding.bind_ortvalue_input("hidden_att_c", hidden_att_c_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm", hidden_lstm_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm_c", hidden_lstm_c_ortvalue)

            io_binding.bind_output("output", "cpu")
            io_binding.bind_output("gate", "cpu")
            io_binding.bind_ortvalue_output("hidden_att_o", hidden_att_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_att_o_c", hidden_att_o_c_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o", hidden_lstm_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o_c", hidden_lstm_o_c_ortvalue)

            self.backward_flow.run_with_iobinding(io_binding)

            outp = io_binding.get_outputs()
            gates = outp[1].numpy()
            residual_outp = [outp[0]] + residual_outp
            if (gates > self.gate_threshold).any():
                break

            # Switch input and output to use latest output as input
            (hidden_att_ortvalue, hidden_att_o_ortvalue) = (
                hidden_att_o_ortvalue,
                hidden_att_ortvalue,
            )
            (hidden_att_c_ortvalue, hidden_att_o_c_ortvalue) = (
                hidden_att_o_c_ortvalue,
                hidden_att_c_ortvalue,
            )
            (hidden_lstm_ortvalue, hidden_lstm_o_ortvalue) = (
                hidden_lstm_o_ortvalue,
                hidden_lstm_ortvalue,
            )
            (hidden_lstm_c_ortvalue, hidden_lstm_o_c_ortvalue) = (
                hidden_lstm_o_c_ortvalue,
                hidden_lstm_c_ortvalue,
            )

        residual = np.concatenate(
            [residual_ort.numpy() for residual_ort in residual_outp], axis=0
        )

        return residual

    def _run_forward_flow(self, residual, enc_outps_ortvalue, num_split):

        residual_o, hidden_att, hidden_lstm = self._init_states(residual)

        hidden_att_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        hidden_att_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        residual_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            residual_o, "cpu", 0
        )

        residual_outp = [residual_ortvalue]
        last_output = residual_ortvalue
        for i in range(residual.shape[0]):

            io_binding = self.forward_flow.io_binding()

            io_binding.bind_cpu_input("residual", residual[i])

            io_binding.bind_ortvalue_input("text", enc_outps_ortvalue)
            io_binding.bind_ortvalue_input("last_output", last_output)

            io_binding.bind_ortvalue_input("hidden_att", hidden_att_ortvalue)
            io_binding.bind_ortvalue_input("hidden_att_c", hidden_att_c_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm", hidden_lstm_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm_c", hidden_lstm_c_ortvalue)

            io_binding.bind_output("output", "cpu")
            io_binding.bind_output("gate", "cpu")
            io_binding.bind_ortvalue_output("hidden_att_o", hidden_att_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_att_o_c", hidden_att_o_c_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o", hidden_lstm_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o_c", hidden_lstm_o_c_ortvalue)

            self.forward_flow.run_with_iobinding(io_binding)

            outp = io_binding.get_outputs()
            gates = outp[1].numpy()
            residual_outp.append(outp[0])
            last_output = outp[0]
            if (gates > self.gate_threshold).any():
                break

            # Switch input and output to use latest output as input
            (hidden_att_ortvalue, hidden_att_o_ortvalue) = (
                hidden_att_o_ortvalue,
                hidden_att_ortvalue,
            )
            (hidden_att_c_ortvalue, hidden_att_o_c_ortvalue) = (
                hidden_att_o_c_ortvalue,
                hidden_att_c_ortvalue,
            )
            (hidden_lstm_ortvalue, hidden_lstm_o_ortvalue) = (
                hidden_lstm_o_ortvalue,
                hidden_lstm_ortvalue,
            )
            (hidden_lstm_c_ortvalue, hidden_lstm_o_c_ortvalue) = (
                hidden_lstm_o_c_ortvalue,
                hidden_lstm_c_ortvalue,
            )
            if len(residual_outp) % num_split == 0 and i != 0:

                residual_o = np.concatenate(
                    [residual_ort.numpy() for residual_ort in residual_outp], axis=0
                )

                yield residual_o
                residual_outp = []
        if len(residual_outp) > 0:
            residual_o = np.concatenate(
                [residual_ort.numpy() for residual_ort in residual_outp], axis=0
            )

            yield residual_o

    def _init_states(self, residual):
        last_outputs = np.zeros(
            [1, residual.shape[1], residual.shape[2]], dtype=np.float32
        )
        hidden_att = [
            np.zeros([1, 1, 1024], dtype=np.float32),
            np.zeros([1, 1, 1024], dtype=np.float32),
        ]
        hidden_lstm = [
            np.zeros([2, 1, 1024], dtype=np.float32),
            np.zeros([2, 1, 1024], dtype=np.float32),
        ]
        return last_outputs, hidden_att, hidden_lstm
