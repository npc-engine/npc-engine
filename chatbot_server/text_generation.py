from typing import List
import numpy as np
import onnxruntime as rt
from chatbot_server.tokenization.tokenization_gpt2 import GPT2TokenizerFast
import os


class GPTTextGenerator:
    def __init__(
        self, gpt_path, max_steps=25, min_length=2, repetition_penalty=1
    ):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(gpt_path, "gpt2.onnx"),
            providers=[rt.get_available_providers()[1]],
            sess_options=sess_options
        )
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt_path, fast=True)
        self.histories = {}
        self.personas = {}
        self.max_steps = max_steps
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

    def add_speaker(self, speaker_id: int, persona: str) -> int:
        self.personas[speaker_id] = persona
        self.histories[speaker_id] = []

    def step_dialog(self, speaker_id: int, line: str, response=None):
        self.histories[speaker_id].append(line)
        if response is None:
            response = self._generate_response(
                self.personas[speaker_id], self.histories[speaker_id]
            )
        self.histories[speaker_id].append(response)
        return response

    def _generate_response(self, persona: str, history: List[str]):
        history = [persona] + history
        utterance = []
        token = None
        for i in range(self.max_steps):

            total = history
            total = "<|endoftext|>".join(total) + "<|endoftext|>"
            ids = self.tokenizer.encode(total)
            ids += utterance
            ids = np.asarray(ids).astype(np.int64).reshape([1, -1])
            o = self.model.run(None, {'input_ids': ids})
            logits = o[0][:, -1, :]
            if i < self.min_length:
                logits[:, -1] = -float("inf")
            token = np.argmax(logits, axis=-1)[0]
            if token == self.tokenizer.eos_token_id:
                break
            utterance += [token]
        return self.tokenizer.decode(utterance)

    def delete_speaker(self, speaker_id: int):
        del self.personas[speaker_id]
        del self.histories[speaker_id]