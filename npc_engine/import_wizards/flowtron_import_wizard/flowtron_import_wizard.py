"""ImportWizard for the Flowtron architecture inference."""
import sys
import os
import json
import yaml
import click
import torch
import numpy as np

from scipy.io.wavfile import write
from npc_engine.import_wizards.base_import_wizard import ImportWizard
from npc_engine.services.tts.flowtron.text import text_to_sequence


def try_to_cuda(tensor):
    """Try to cuda."""
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


class FlowtronImportWizard(ImportWizard):
    """ImportWizard for the Flowtron architecture inference.

    Paper:
    [arXiv:2005.05957](https://arxiv.org/abs/2005.05957)
    Code:
    https://github.com/NVIDIA/flowtron
    """

    def convert(self, model_path: str, export_path: str):
        """Convert the model to the desired format.

        Args:
            model_path: Path to the model.
            export_path: Path to the exported model.
        """
        flowtron_path = os.path.join(model_path, "flowtron.pt")
        if not os.path.exists(flowtron_path):
            click.echo(
                click.style(
                    "Flowtron model checkpoint should be present and named `flowtron.pt`.",
                    fg="red",
                )
            )
        waveglow_path = os.path.join(model_path, "waveglow.pt")
        if not os.path.exists(waveglow_path):
            click.echo(
                click.style(
                    "Waveglow model checkpoint should be present and named `waveglow.pt`.",
                    fg="red",
                )
            )
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            click.echo(
                click.style(
                    "Config  should be present and named `config.json`.", fg="red"
                )
            )
        # check if flowtron was cloned
        if not os.path.exists(os.path.join(model_path, "flowtron", "flowtron.py")):
            click.echo(
                click.style(
                    "Please set up `https://github.com/NVIDIA/flowtron.git`"
                    + " following README instructions in the model folder.",
                    fg="red",
                )
            )
        # Append flowtron, tacotron and waveglow repositories to PYTHONPATH
        sys.path.append(os.path.join(model_path, "flowtron"))
        sys.path.append(os.path.join(model_path, "flowtron", "tacotron2"))
        sys.path.append(os.path.join(model_path, "flowtron", "tacotron2", "waveglow"))

        from .flowtron_onnx import Flowtron, FlowtronTTS, FlowtronEncoder, init_states
        from glow import WaveGlow  # noqa: F401, E261

        text = """
        I am doing fine
        """

        # load waveglow
        waveglow = try_to_cuda(torch.load(waveglow_path)["model"]).eval()
        for k in waveglow.convinv:
            k.float()
        waveglow.eval()

        with open(config_path, "r") as f:
            model_config = json.load(f)

        # load flowtron
        model = try_to_cuda(Flowtron(**model_config["model_config"]))
        try:
            state_dict = torch.load(flowtron_path, map_location="cpu")["state_dict"]
        except KeyError:
            state_dict = torch.load(flowtron_path, map_location="cpu")[
                "model"
            ].state_dict()
        model.load_state_dict(state_dict, False)
        model.eval()

        # Script loop parts of the flows
        # model.script_flows()

        n_frames_str = click.prompt("Maximum sequence length", default=512)
        sigma = 0.8
        try:
            self.n_frames = int(n_frames_str)
        except Exception:
            self.n_frames = 512

        speaker_vecs = try_to_cuda(torch.zeros([1], dtype=torch.long))
        text = try_to_cuda(torch.LongTensor(text_to_sequence(text)).view([1, -1]))
        with torch.no_grad():
            residual = try_to_cuda(
                torch.FloatTensor(1, 80, self.n_frames).normal_() * sigma
            )

            encoder = FlowtronEncoder(
                model.embedding, model.speaker_embedding, model.encoder
            )
            waveglow = FlowtronTTS.patch_waveglow(waveglow)
            model = FlowtronTTS(encoder, model, waveglow)
            text = text.reshape([1, -1])
            enc_outps = encoder(speaker_vecs, text)
            torch.onnx.export(
                encoder,
                (speaker_vecs, text),
                os.path.join(export_path, "encoder.onnx"),
                opset_version=11,
                do_constant_folding=True,
                input_names=["speaker_vecs", "text"],
                output_names=["text_emb"],
                dynamic_axes={"text": {1: "text_seq"}, "text_emb": {0: "text_seq"}},
                verbose=False,
            )

            backward_flow = model.backward_flow.ar_step
            residual = residual.permute(2, 0, 1)
            residual_o, hidden_att, hidden_lstm = init_states(residual)
            residual_o = try_to_cuda(residual_o)
            hidden_att = try_to_cuda(hidden_att)
            hidden_lstm = try_to_cuda(hidden_lstm)

            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = backward_flow(
                residual[0],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            torch.onnx.export(
                backward_flow,
                (
                    residual[0],
                    enc_outps,
                    residual_o,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                os.path.join(export_path, "backward_flow.onnx"),
                opset_version=11,
                do_constant_folding=True,
                input_names=[
                    "residual",
                    "text",
                    "last_output",
                    "hidden_att",
                    "hidden_att_c",
                    "hidden_lstm",
                    "hidden_lstm_c",
                ],
                output_names=[
                    "output",
                    "gate",
                    "hidden_att_o",
                    "hidden_att_o_c",
                    "hidden_lstm_o",
                    "hidden_lstm_o_c",
                ],
                dynamic_axes={"text": {0: "text_seq"}},
                verbose=False,
            )

            forward_flow = model.forward_flow

            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = forward_flow(
                residual[0],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            torch.onnx.export(
                forward_flow,
                (
                    residual[0],
                    enc_outps,
                    residual_o,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                os.path.join(export_path, "forward_flow.onnx"),
                opset_version=11,
                do_constant_folding=True,
                input_names=[
                    "residual",
                    "text",
                    "last_output",
                    "hidden_att",
                    "hidden_att_c",
                    "hidden_lstm",
                    "hidden_lstm_c",
                ],
                output_names=[
                    "output",
                    "gate",
                    "hidden_att_o",
                    "hidden_att_o_c",
                    "hidden_lstm_o",
                    "hidden_lstm_o_c",
                ],
                dynamic_axes={"text": {0: "text_seq"}},
                verbose=False,
            )

            residual = residual.permute(1, 2, 0)
            mels = model(residual, speaker_vecs, text)

            audio = waveglow(mels, sigma=0.8)
            print(f"audio max {audio.max()} min {audio.min()} shape {audio.shape}")
            write(
                os.path.join(export_path, "audio.wav"),
                model_config["data_config"]["sampling_rate"],
                audio.to("cpu").numpy().astype(np.float32).reshape([-1, 1]),
            )

            torch.onnx.export(
                waveglow,
                (mels),
                os.path.join(export_path, "waveglow.onnx"),
                opset_version=11,
                do_constant_folding=True,
                input_names=["mels"],
                output_names=["audio"],
                dynamic_axes={"mels": {2: "mel_seq"}, "audio": {1: "audio_seq"}},
                verbose=False,
            )

    def create_config(self, export_path: str):
        """Create the config for the model.

        Args:
            export_path: Path to the exported model.
        """
        if hasattr(self, "n_frames"):
            n_frames = self.n_frames
        else:
            n_frames_str = click.prompt(
                "Number of frames in the input spectrogram", default=512
            )
            n_frames = int(n_frames_str)

        gate_threshold = click.prompt("Gate threshold", default=0.5)
        gate_threshold = float(gate_threshold)
        sigma = click.prompt("Sigma", default=0.8)
        sigma = float(sigma)
        smoothing_window = click.prompt("Smoothing window", default=3)
        smoothing_window = int(smoothing_window)
        smoothing_weight = click.prompt("Smoothing weight", default=0.5)
        smoothing_weight = float(smoothing_weight)

        config = {
            "max_frames": n_frames,
            "gate_threshold": gate_threshold,
            "sigma": sigma,
            "smoothing_window": smoothing_window,
            "smoothing_weight": smoothing_weight,
        }
        with open(os.path.join(export_path, "config.yml"), "w") as f:
            yaml.dump(config, f)

    @classmethod
    def get_api(cls) -> str:
        """Get the api for the exporter."""
        return "TextToSpeechAPI"

    @classmethod
    def get_model_name(cls) -> str:
        """Get the model name."""
        return "ESPNetTTS"

    def test_model_impl(self):
        """Test the model implementation."""
        pass
