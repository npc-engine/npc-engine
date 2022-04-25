"""Module that implements Huggingface transformers semantic similarity."""
from typing import List
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from tokenizers import Tokenizer
from npc_engine.models.similarity.similarity_base import SimilarityAPI
import os
from scipy.spatial.distance import cdist


class TransformerSemanticSimilarity(SimilarityAPI):
    """Huggingface transformers semantic similarity.

    Uses ONNX export of Huggingface transformers
    (https://huggingface.co/models) with biencoder architecture.
    """

    def __init__(
        self,
        model_path: str,
        metric: str = "dot",
        pad_token_id: int = 0,
        *args,
        **kwargs
    ):
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
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options,
        )

        self.pad_token_id = pad_token_id
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
        self.metric_type = metric

    def compute_embedding(self, line: str) -> np.ndarray:
        """Compute line embeddings in batch.

        Args:
            lines: List of sentences to embed

        Returns:
            Embedding batch of shape (batch_size, embedding_size)
        """
        ids = (
            np.asarray(self.tokenizer.encode(line).ids)
            .reshape([1, -1])
            .astype(np.int64)
        )
        attention_mask = np.ones_like(ids)
        outp = self.model.run(
            None, {"input_ids": ids, "attention_mask": attention_mask}
        )
        return self._mean_pooling(outp, attention_mask)

    def compute_embedding_batch(self, lines: List[str]) -> np.ndarray:
        """Compute sentence embedding.

        Args:
            line: Sentence to embed

        Returns:
            Embedding of shape (1, embedding_size)
        """
        tokenized = self.tokenizer.encode_batch(lines)
        ids = np.stack([np.asarray(encoding.ids) for encoding in tokenized]).astype(
            np.int64
        )
        attention_mask = np.stack(
            [np.asarray(encoding.attention_mask) for encoding in tokenized]
        ).astype(np.int64)
        outp = self.model.run(
            None, {"input_ids": ids, "attention_mask": attention_mask}
        )
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
            return -np.dot(embedding_a, embedding_b.T).squeeze()
        elif self.metric_type == "cosine":
            return 1 - cdist(embedding_a, embedding_b, metric="cosine").squeeze()
