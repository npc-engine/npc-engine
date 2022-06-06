"""Module that implements Huggingface transformers semantic similarity."""
from typing import List
import json
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from tokenizers import Tokenizer
from npc_engine.services.similarity.similarity_base import SimilarityAPI
import os
from scipy.spatial.distance import cdist


class TransformerSemanticSimilarity(SimilarityAPI):
    """Huggingface transformers semantic similarity.

    Uses ONNX export of Huggingface transformers
    (https://huggingface.co/models) with biencoder architecture.
    Also requires a tokenizer.json with huggingface tokenizers definition.

    model.onnx spec:

        - inputs:
            `input_ids` of shape `(batch_size, sequence)`
            `attention_mask` of shape `(batch_size, sequence)`
            (Optional) `input_type_ids` of shape `(batch_size, sequence)`
        - outputs:
            `token_embeddings` of shape `(batch_size, sequence, hidden_size)`
    """

    def __init__(self, model_path: str, metric: str = "dot", *args, **kwargs):
        """Create and load biencoder model for semantic similarity.

        Args:
            model_path: A path where model config and weights are
            metric: distance to compute semantic similarity
        """
        super().__init__(*args, **kwargs)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        input_names = [inp.name for inp in self.model.get_inputs()]
        self.token_type_support = "token_type_ids" in input_names
        special_tokens_map_path = os.path.join(model_path, "special_tokens_map.json")
        with open(special_tokens_map_path, "r") as f:
            self.special_tokens = json.load(f)

        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        self.pad_token_id = self.tokenizer.encode(self.special_tokens["pad_token"]).ids[
            0
        ]
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
        self.metric_type = metric

    def compute_embedding(self, line: str) -> np.ndarray:
        """Compute sentence embedding.

        Args:
            line: Sentence to embed

        Returns:
            Embedding of shape (1, embedding_size)
        """
        ids = (
            np.asarray(self.tokenizer.encode(line).ids)
            .reshape([1, -1])
            .astype(np.int64)
        )
        attention_mask = np.ones_like(ids)
        if not self.token_type_support:
            input_dict = {"input_ids": ids, "attention_mask": attention_mask}
        else:
            input_dict = {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros_like(ids),
            }
        outp = self.model.run(None, input_dict)
        return self._mean_pooling(outp, attention_mask)

    def compute_embedding_batch(self, lines: List[str]) -> np.ndarray:
        """Compute line embeddings in batch.

        Args:
            lines: List of sentences to embed

        Returns:
            Embedding batch of shape (batch_size, embedding_size)
        """
        tokenized = self.tokenizer.encode_batch(lines)
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
        return self._mean_pooling(outp, attention_mask)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        attention_mask = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * attention_mask, 1)
        sum_mask = np.clip(attention_mask.sum(1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def metric(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> np.ndarray:
        """Similarity between two embeddings.

        Embeddings are of broadcastable shapes. (1 or batch_size)
        Args:
            embedding_a: Embedding of shape (1 or batch_size, embedding_size)
            embedding_b: Embedding of shape (1 or batch_size, embedding_size)

        Returns:
            Vector of distances (batch_size or 1,)
        """
        if self.metric_type == "dot":
            return -np.dot(embedding_a, embedding_b.T).squeeze(0)
        elif self.metric_type == "cosine":
            return 1 - cdist(embedding_a, embedding_b, metric="cosine").squeeze(0)
