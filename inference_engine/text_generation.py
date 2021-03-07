from typing import List
import numpy as np
import scipy.special as scp
import onnxruntime as rt
from inference_engine.tokenization.tokenization_roberta import RobertaTokenizerFast
import os


class BartChatbot:
    def __init__(self, model_path, max_steps=100, min_length=2, repetition_penalty=1):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "bart.onnx"),
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options,
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, fast=True)
        self.max_steps = max_steps
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

    def generate_reply(
        self, persona: str, history: List[str], temperature: float, topk: int = None
    ):
        persona = self.tokenizer(persona).input_ids

        history = [
            self.tokenizer(turn, add_special_tokens=False).input_ids
            + [self.tokenizer.eos_token_id]
            for turn in history
        ]
        total = persona
        for turn in history:
            total += turn
        total = np.asarray(total)

        utterance = np.asarray([self.tokenizer.eos_token_id], dtype=np.int64).reshape(
            [1, 1]
        )

        for i in range(self.max_steps):
            o = self.model.run(
                None, {"input_ids": total, "decoder_input_ids": utterance}
            )
            logits = o[0][0, -1, :]

            if topk is not None:
                ind = np.argpartition(logits, -topk)[-topk:]
                logits = logits[ind]

            probs = scp.softmax(logits / temperature, axis=0)

            if i < self.min_length and topk is None:
                probs += probs[self.tokenizer.eos_token_id] / (probs.shape[0] - 1)
                probs[self.tokenizer.eos_token_id] = 0.0
            elif i < self.min_length:
                indices = np.nonzero(ind == self.tokenizer.eos_token_id)
                if indices[0].size > 0:
                    probs += probs[indices[0][0]] / (probs.shape[0] - 1)
                    probs[indices[0][0]] = 0.0

            token = np.random.choice(np.arange(probs.shape[0]), p=probs)
            # token = np.argmax(logits, axis=-1)[0]
            token = token.reshape([1, 1])
            utterance = np.concatenate([utterance, token], axis=1)
            if token[0, 0] == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.batch_decode(
            utterance.tolist(), skip_special_tokens=True
        )[0]
