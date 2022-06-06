"""Module that implements Huggingface transformers classification."""
import os
import json
from typing import List, Tuple, Union
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from tokenizers import Tokenizer
from npc_engine.services.sequence_classifier.sequence_classifier_base import (
    SequenceClassifierAPI,
)


class HfClassifier(SequenceClassifierAPI):
    """Huggingface transformers sequence classification.

    Uses ONNX export of Huggingface transformers
    (https://huggingface.co/models) with sequence-classification feature.
    Also requires saved tokenizer with huggingface tokenizers.
    """

    def __init__(self, model_path: str, *args, **kwargs):
        """Create and load biencoder model for semantic similarity.

        Args:
            model_path: A path where model config and weights are.
        """
        super().__init__(*args, **kwargs)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        input_names = [inp.name for inp in self.model.get_inputs()]
        self.token_type_support = "token_type_ids" in input_names
        special_tokens_map_path = os.path.join(model_path, "special_tokens_map.json")
        with open(special_tokens_map_path, "r") as f:
            self.special_tokens = json.load(f)

        self.pad_token_id = self.tokenizer.encode(self.special_tokens["pad_token"]).ids[
            0
        ]
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        self.tokenizer.enable_padding(
            direction="right",
            pad_id=self.pad_token_id,
            pad_type_id=0,
            pad_token=self.tokenizer.decode(
                [self.pad_token_id], skip_special_tokens=False
            ),
            length=None,
            pad_to_multiple_of=None,
        )
        self.tests = {}

    def compute_scores_batch(
        self, texts: List[Union[str, Tuple[str, str]]]
    ) -> np.ndarray:
        """Compute scores.

        Args:
            texts: Sentence to embed

        Returns:
            scores: Scores for each text
        """
        tokenized = self.tokenizer.encode_batch(texts)
        ids = np.stack([np.asarray(encoding.ids) for encoding in tokenized]).astype(
            np.int64
        )
        attention_mask = np.stack(
            [np.asarray(encoding.attention_mask) for encoding in tokenized]
        ).astype(np.int64)
        if not self.token_type_support:
            input_dict = {"input_ids": ids, "attention_mask": attention_mask}
        else:
            input_dict = {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.stack(
                    [np.asarray(encoding.type_ids) for encoding in tokenized]
                ).astype(np.int64),
            }
        outp = self.model.run(None, input_dict)
        return outp[0]
