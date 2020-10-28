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

    def add_speaker(self, speaker_id: str, persona: str) -> int:
        self.personas[speaker_id] = self.tokenizer.encode(
            persona + self.tokenizer.eos_token
        )
        self.histories[speaker_id] = []

    def step_dialog(self, speaker_id: str, line: str, response=None):
        self.histories[speaker_id].append(
            self.tokenizer.encode(line + self.tokenizer.eos_token)
        )
        print(self.histories[speaker_id][-1])
        if response is None:
            response = self._generate_response(
                self.personas[speaker_id], self.histories[speaker_id]
            )
            self.histories[speaker_id].append(response)
            return self.tokenizer.decode(response, skip_special_tokens=True)
        else:
            self.histories[speaker_id].append(
                self.tokenizer.encode(response + self.tokenizer.eos_token)
            )
            return response

    def _generate_response(self, persona: List[int], history: List[List[int]]):
        utterance = []
        token = None
        total = [persona] + history
        total_token_types = np.concatenate([
            np.zeros([1, len(step)], dtype=np.int64) if (i % 2) == 0 else
            np.ones([1, len(step)], dtype=np.int64)
            for i, step in enumerate(total)
        ], axis=1)

        ids_list = []
        for el in total:
            ids_list += el
        for i in range(self.max_steps):
            ids = np.asarray(
                ids_list + utterance
            ).astype(np.int64).reshape([1, -1])
            o = self.model.run(None, {
                'input_ids': ids, 'token_type_ids': total_token_types
            })
            logits = o[0][:, -1, :]
            if i < self.min_length:
                logits[:, -1] = -float("inf")
            token = np.argmax(logits, axis=-1)[0]
            utterance += [token]
            if token == self.tokenizer.eos_token_id:
                break
            total_token_types = np.concatenate(
                [total_token_types, np.zeros([1, 1], dtype=np.int64)], axis=1
            )
        return utterance

    def delete_speaker(self, speaker_id: str):
        del self.personas[speaker_id]
        del self.histories[speaker_id]

    def empty_history(self, speaker_id: str):
        self.histories[speaker_id] = []
