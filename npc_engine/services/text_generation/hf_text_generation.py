"""BART based chatbot implementation."""
from typing import Any, Dict
import numpy as np
import scipy.special as scp
import onnxruntime as rt
from npc_engine.services.text_generation.text_generation_base import TextGenerationAPI
from tokenizers import Tokenizer
import os
import json
from npc_engine.services.utils import DTYPE_MAP


class HfChatbot(TextGenerationAPI):
    """Chatbot that uses Huggingface transformer architectures.

    ONNX export of Huggingface transformer is required (see https://huggingface.co/docs/transformers/serialization).
    Features seq2seq-lm, causal-lm, seq2seq-lm-with-past, causal-lm-with-past are supported
    """

    def __init__(
        self,
        model_path,
        max_length=100,
        min_length=2,
        repetition_penalty=1,
        *args,
        **kwargs,
    ):
        """Create the chatbot from config args and kwargs.

        Args:
            model_path: path to scan for model files (weights and configs)
            max_length: stop generation at this number of tokens
            min_length: model can't stop generating text before it's atleast
                this long in tokens
            repetition_penalty: probability coef for same tokens to appear multiple times

        """
        super().__init__(*args, **kwargs)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

        self.max_steps = max_length
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty

        special_tokens_map_path = os.path.join(model_path, "special_tokens_map.json")
        with open(special_tokens_map_path, "r") as f:
            self.special_tokens = json.load(f)

        self.eos_token_id = self.tokenizer.encode(self.special_tokens["eos_token"]).ids[
            0
        ]

        self.model_inputs = self.model.get_inputs()
        self.is_encdec = (
            len([i.name for i in self.model_inputs if "decoder" in i.name]) > 0
        )
        self.with_past = (
            len([i.name for i in self.model_inputs if "past_key_values" in i.name]) > 0
        )
        self.shape_dict = {
            "batch": 1,
            "past_encoder_sequence": 0,
            "past_decoder_sequence": 0,
            "past_sequence + sequence": 0,
        }
        self.dtypes = {i.name: DTYPE_MAP[i.type] for i in self.model_inputs}

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
        inputs = self.create_starter_inputs(prompt)
        utterance = []
        for i in range(self.max_steps):
            o = self.model.run(
                None,
                inputs,
            )
            logit = o[0][0, -1, :]
            if i < self.min_length:
                logit[self.eos_token_id] = float("-inf")
            token = self.decode_logit(logit, temperature, topk)
            utterance.append(token)
            result_dict = {
                outp.name: o[i] for i, outp in enumerate(self.model.get_outputs())
            }
            inputs = self.update_inputs_with_results(inputs, result_dict, token)
            if token == self.eos_token_id:
                break
        return self.tokenizer.decode(utterance, skip_special_tokens=True)

    def create_starter_inputs(self, prompt: str = "") -> Dict[str, Any]:
        """Create starter inputs for the model.

        Args:
            prompt: Prompt to start generation from.

        Returns:
            Dict of inputs to the model
        """
        tokens = self.tokenizer.encode(prompt).ids
        inputs = {}
        if self.is_encdec:
            prompt_start = tokens[-1:]
            inputs["input_ids"] = np.asarray(
                tokens[:-1], dtype=self.dtypes["input_ids"]
            ).reshape([1, -1])
            inputs["decoder_input_ids"] = np.asarray(
                prompt_start, dtype=self.dtypes["decoder_input_ids"]
            ).reshape([1, -1])
            inputs["attention_mask"] = np.ones(
                [1, 6], dtype=self.dtypes["attention_mask"]
            )
            inputs["decoder_attention_mask"] = np.ones(
                [1, 3], dtype=self.dtypes["decoder_attention_mask"]
            )
        else:
            inputs["input_ids"] = np.asarray(
                tokens, dtype=self.dtypes["input_ids"]
            ).reshape([1, -1])
            inputs["attention_mask"] = np.ones_like(
                inputs["input_ids"], dtype=self.dtypes["attention_mask"]
            )

        if self.with_past:
            for i in self.model_inputs:
                if "past_key_values" in i.name:
                    shape_tuple = [self.shape_dict.get(dim, dim) for dim in i.shape]
                    inputs[i.name] = np.empty(shape_tuple, dtype=self.dtypes[i.name])
        return inputs

    def decode_logit(self, logit: np.ndarray, temperature: float, topk: int) -> int:
        """Decode logit to token.

        Args:
            logit: Logit to decode of shape (vocab_size,)

        Returns:
            Decoded token of shape
        """
        if topk is not None:
            ind = np.argpartition(logit, -topk)[-topk:]
            new_logits = np.zeros(logit.shape)
            new_logits[ind] = logit[ind]
            logit = new_logits

        probs = scp.softmax(logit / temperature, axis=0)
        token = np.random.choice(np.arange(probs.shape[0]), p=probs)
        token = token.ravel()[0]
        return token

    def update_inputs_with_results(
        self,
        inputs: Dict[str, np.ndarray],
        results: Dict[str, np.ndarray],
        decoded_token: int,
    ) -> Dict[str, np.ndarray]:
        """Update inputs with results from model.

        Args:
            inputs: Inputs to the model
            results: Results from the model

        Returns:
            Updated inputs
            Finished generation
        """
        ids_name = "decoder_input_ids" if self.is_encdec else "input_ids"
        att_mask_name = "decoder_attention_mask" if self.is_encdec else "attention_mask"

        if self.with_past:
            inputs[ids_name] = np.asarray(
                [decoded_token], dtype=self.dtypes[ids_name]
            ).reshape([1, -1])
            inputs[att_mask_name] = np.ones(
                [1, inputs[att_mask_name].shape[-1] + 1],
                dtype=self.dtypes[att_mask_name],
            )
            for inp in self.model_inputs:
                if "past_key_values" in inp.name:
                    inputs[inp.name] = results[
                        inp.name.replace("past_key_values", "present")
                    ]
            if self.is_encdec:
                inputs.pop("input_ids", None)
        else:
            decoder_input_ids = inputs[ids_name]
            decoder_attention_mask = inputs[att_mask_name]
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, np.asarray([[decoded_token]], dtype=np.int32)],
                axis=1,
            )
            decoder_attention_mask = np.ones_like(
                decoder_input_ids, dtype=self.dtypes[att_mask_name]
            )
            inputs[ids_name] = decoder_input_ids
            inputs[att_mask_name] = decoder_attention_mask
        return inputs

    def get_special_tokens(self) -> Dict[str, str]:
        """Retrun dict of special tokens to be renderable from template."""
        return self.special_tokens
