from typing import Dict, Any
import numpy as np
import scipy.special as scp
import onnxruntime as rt
from inference_engine.models.chatbot.chatbot_base import Chatbot
from inference_engine.tokenization.tokenization_roberta import RobertaTokenizerFast
import os


class BartChatbot(Chatbot):
    def __init__(
        self,
        model_path,
        max_steps=100,
        min_length=2,
        repetition_penalty=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self.encoder_model = rt.InferenceSession(
            os.path.join(model_path, "encoder_bart.onnx"),
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options,
        )
        self.decoder_model = rt.InferenceSession(
            os.path.join(model_path, "decoder_bart.onnx"),
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options,
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, fast=True)
        self.max_steps = max_steps
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

    def run(self, prompt: str, temperature: float, topk: int = None):
        print(prompt)
        tokens = self.tokenizer(prompt)
        total = np.asarray(tokens, dtype=np.int64).reshape([1, -1])
        print(total.shape)
        total_enc = self.encoder_model.run(None, {"input_ids": total})[0]

        utterance = np.asarray([self.tokenizer.eos_token_id], dtype=np.int64).reshape(
            [1, 1]
        )

        for i in range(self.max_steps):
            o = self.decoder_model.run(
                None,
                {"encoder_hidden_state": total_enc, "decoder_input_ids": utterance},
            )
            logits = o[0][0, -1, :]

            if i < self.min_length:
                logits[self.tokenizer.eos_token_id] = float("-inf")
            if topk is not None:
                ind = np.argpartition(logits, -topk)[-topk:]
                new_logits = np.zeros(logits.shape)
                new_logits[ind] = logits[ind]
                logits = new_logits

            probs = scp.softmax(logits / temperature, axis=0)

            token = np.random.choice(np.arange(probs.shape[0]), p=probs)
            token = token.reshape([1, 1])
            utterance = np.concatenate([utterance, token], axis=1)
            if token[0, 0] == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.batch_decode(
            utterance.tolist(), skip_special_tokens=True
        )[0]
