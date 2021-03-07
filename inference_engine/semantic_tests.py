from typing import Union, Dict, Tuple, List
from collections import namedtuple
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from inference_engine.tokenization.tokenization_roberta import RobertaTokenizerFast
import os
from scipy.spatial.distance import cosine
import logging

DialogLine = namedtuple("DialogLine", "cue_emb response expires_after threshold")


class SemanticTests:
    def __init__(self, model_path):
        self.OR_METHOD = "or"
        self.AND_METHOD = "and"
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options,
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, fast=True)
        self.tests = {}

    def add_test(self, test_id, test_strings):
        self.tests[test_id] = [
            self._compute_embedding(test_string) for test_string in test_strings
        ]

    def test(self, line, tests, method: str):
        line_embedding = self._compute_embedding(line)
        results = {}
        for test_id in tests:
            try:
                results[test_id] = self._test(
                    line_embedding, self.tests[test_id], method
                )
            except Exception as e:
                logging.error("{} test failed to compute {}".format(test_id, e))
        return results

    def test_custom(self, line, query_lines, method: str):
        line_embedding = self._compute_embedding(line)
        query_embeddings = self._compute_embedding_batch(query_lines)
        try:
            return self._test(line_embedding, query_embeddings, method)
        except Exception as e:
            logging.error("{} test failed to compute {}".format("Custom", e))
        return -1

    def _test(self, line_embedding, query_embeddings, method):
        if method == self.AND_METHOD:
            mean_emb = query_embeddings.mean(axis=0)
            return cosine(line_embedding, mean_emb)
        elif method == self.OR_METHOD:
            return cosine(line_embedding, query_embeddings).max()
        return -1

    def _compute_embedding(self, line: str) -> np.ndarray:
        ids = np.asarray(self.tokenizer.encode(line)).reshape([1, -1]).astype(np.int64)
        attention_mask = np.ones_like(ids)
        outp = self.model.run(
            None, {"input_ids": ids, "attention_mask": attention_mask}
        )
        return outp[1]

    def _compute_embedding_batch(self, lines: List[str]) -> np.ndarray:
        tokenized = self.tokenizer(lines)
        ids = np.asarray(tokenized.input_ids).astype(np.int64)
        attention_mask = np.asarray(tokenized.attention_mask).astype(np.int64)
        outp = self.model.run(
            None, {"input_ids": ids, "attention_mask": attention_mask}
        )
        return outp[1]
