"""BART based chatbot implementation."""
from typing import Dict
import numpy as np
import scipy.special as scp
import onnxruntime as rt
from npc_engine.services.text_generation.text_generation_base import TextGenerationAPI
from tokenizers import Tokenizer
import os
import json


class BartChatbot(TextGenerationAPI):
    """BART based chatbot implementation class.

    This model class requires two ONNX models `encoder_bart.onnx` and `decoder_bart.onnx`
    that correspond to encoder and decoder from transformers
    [EncoderDecoderModel](https://huggingface.co/transformers/model_doc/encoderdecoder.html)
    and a tokenizer.json with huggingface tokenizers definition.

    encoder_bart.onnx spec:

        - inputs:
            `input_ids`
        - outputs:
            `encoder_hidden_state`

    decoder_bart.onnx spec:

        - inputs:
            `encoder_hidden_state`
            `decoder_input_ids`
        - outputs:
            `logits`
    """

    def __init__(
        self,
        model_path,
        max_steps=100,
        min_length=2,
        repetition_penalty=1,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        sep_token_id=None,
        *args,
        **kwargs,
    ):
        """Create the chatbot from config args and kwargs.

        Args:
            model_path: path to scan for model files (weights and configs)
            max_steps: stop generation at this number of tokens
            min_length: model can't stop generating text before it's atleast
                this long in tokens
            repetition_penalty: probability coef for same tokens to appear multiple times
            bos_token_id: beginning of sequence token id
            eos_token_id: end of sequence token id
            pad_token_id: padding token id
            sep_token_id: token id for separating sequence into multiple parts

        """
        super().__init__(*args, **kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = eos_token_id if sep_token_id is None else sep_token_id
        self.pad_token_id = pad_token_id

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        self.encoder_model = rt.InferenceSession(
            os.path.join(model_path, "encoder_bart.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.decoder_model = rt.InferenceSession(
            os.path.join(model_path, "decoder_bart.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        added_tokens_path = os.path.join(model_path, "added_tokens.txt")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path) as f:
                added_tokens = json.load(f)
            added_tokens = [
                key for key, _ in sorted(list(added_tokens.items()), key=lambda x: x[1])
            ]
        self.tokenizer.add_tokens(added_tokens)
        self.special_tokens = {
            "bos_token": self.tokenizer.decode(
                [bos_token_id], skip_special_tokens=False
            ),
            "eos_token": self.tokenizer.decode(
                [eos_token_id], skip_special_tokens=False
            ),
            "sep_token": self.tokenizer.decode(
                [self.sep_token_id], skip_special_tokens=False
            ),
            "pad_token": self.tokenizer.decode(
                [pad_token_id], skip_special_tokens=False
            ),
            **{
                f"added_token{self.tokenizer.token_to_id(token)}": token
                for token in added_tokens
            },
        }

        self.max_steps = max_steps
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

    def run(self, prompt: str, temperature: float = 1.0, topk: int = None) -> str:
        """Run text generation from given prompt and parameters.

        Args:
            prompt: Fromatted prompt.
            temperature: Temperature parameter for sampling.
                Controls how random model output is: more temperature - more randomness
            topk: If not none selects top n of predictions to sample from during generation.

        Returns:
            Generated text
        """
        tokens = self.tokenizer.encode(prompt)
        total = np.asarray(tokens.ids, dtype=np.int64).reshape([1, -1])
        total_enc = self.encoder_model.run(None, {"input_ids": total})[0]

        utterance = np.asarray([self.eos_token_id], dtype=np.int64).reshape([1, 1])

        for i in range(self.max_steps):
            o = self.decoder_model.run(
                None,
                {"encoder_hidden_state": total_enc, "decoder_input_ids": utterance},
            )
            logits = o[0][0, -1, :]

            if i < self.min_length:
                logits[self.eos_token_id] = float("-inf")
            if topk is not None:
                ind = np.argpartition(logits, -topk)[-topk:]
                new_logits = np.zeros(logits.shape)
                new_logits[ind] = logits[ind]
                logits = new_logits

            probs = scp.softmax(logits / temperature, axis=0)

            token = np.random.choice(np.arange(probs.shape[0]), p=probs)
            token = token.reshape([1, 1])
            utterance = np.concatenate([utterance, token], axis=1)
            if token[0, 0] == self.eos_token_id:
                break
        return self.tokenizer.decode(utterance[0, :].tolist(), skip_special_tokens=True)

    def get_special_tokens(self) -> Dict[str, str]:
        """Retrun dict of special tokens to be renderable from template."""
        return self.special_tokens
