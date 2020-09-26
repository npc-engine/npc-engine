from typing import List
import numpy as np
import onnxruntime as rt
from chatbot_server.tokenization.tokenization_gpt2 import RobertaTokenizer
import os


class DialogScriptSystem:
    def __init__(
        self, model_path
    ):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=[rt.get_available_providers()[1]],
            sess_options=sess_options
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, fast=True)
        self.histories = {}
        self.personas = {}

    def add_speaker(self, speaker_id: int, persona: str) -> int:
        self.personas[speaker_id] = persona
        self.histories[speaker_id] = []

    def step_dialog(self, speaker_id: int, line: str, response=None):
        pass
        return response

    def script_line(self, speaker_id: int, parent: int, cue_line: str, script_line: str):
        pass

    def delete_speaker(self, speaker_id: int):
        del self.personas[speaker_id]
        del self.histories[speaker_id]