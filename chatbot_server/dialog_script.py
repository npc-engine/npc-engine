from typing import Union, Dict, Tuple
from collections import namedtuple
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from chatbot_server.tokenization.tokenization_roberta import RobertaTokenizerFast
import os
from treelib import Tree
from scipy.spatial.distance import cosine

DialogLine = namedtuple("DialogLine", "cue_emb response expires_after threshold")


class DialogScriptSystem:
    def __init__(
        self, model_path
    ):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            providers=[rt.get_available_providers()[0]],
            sess_options=sess_options
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_path, fast=True
            )
        self.dialog_trees: Dict[str, Tree] = {}
        self.current_nodes: Dict[str, str] = {}
        self.no_activation_cnts: Dict[str, int] = {}

    def add_speaker(self, speaker_id: str):
        self.dialog_trees[speaker_id] = Tree()
        self.dialog_trees[speaker_id].create_node("Root node", "root")
        self.current_nodes[speaker_id] = "root"
        self.no_activation_cnts[speaker_id] = 0

    def step_dialog(self, speaker_id: str, line: str) -> Union[None, Tuple[str, str]]:
        reply = self._get_reply(speaker_id, line)
        if reply is None:
            self.no_activation_cnts[speaker_id] += 1
        node = self.dialog_trees[speaker_id].get_node(
            self.current_nodes[speaker_id]
        )
        if (
            self.current_nodes[speaker_id] != 'root'
            and self.no_activation_cnts[speaker_id] >= node.data.expires_after
        ):
            self.current_nodes[speaker_id] = node.bpointer
        children = self.dialog_trees[speaker_id].children(
            self.current_nodes[speaker_id]
        )
        if not children:
            self.current_nodes[speaker_id] = "root"
        return reply

    def _get_reply(self, speaker_id: str, line: str) -> Union[None, Tuple[str, str]]:
        child_nodes = self.dialog_trees[speaker_id].children(
            self.current_nodes[speaker_id]
        )
        if not child_nodes:
            return None
        line_emb = self._compute_embedding(line)
        similarities = [
            1 - cosine(line_emb, child_node.data.cue_emb)
            for child_node in child_nodes
        ]
        above_threshold = [
            similarities[i] > child_node.data.threshold
            for i, child_node in enumerate(child_nodes)
        ]
        ids = np.argsort(similarities)[::-1]
        reply_id = None
        for idx in ids:
            if above_threshold[idx]:
                reply_id = idx
                break
        if reply_id is None:
            return None
        else:
            self.current_nodes[speaker_id] = child_nodes[reply_id].identifier
            return (
                child_nodes[reply_id].data.response,
                child_nodes[reply_id].identifier
            )

    def script_line(
        self,
        speaker_id: str,
        parent: str,
        node_id: str,
        cue_line: str,
        script_line: str,
        expires_after: int,
        threshold: float,
    ):
        """Add line to scripted dialog engine.

        Line added to the engine will be checked for similarity
        at each dialog step and script_line will be returned if
        similarity threshold is reached.

        Args:
            speaker_id (str): Unique speaker identifier
            parent (str): Parent line identifier, scripted line
                is inactive if parent is not reached
            node_id (str): Unique line identifier to be assigned
            cue_line (str): String to be checked for similarity to user input
            script_line (str): Response that is triggered
                if similarity threshold is reached
            expires_after (int): Number of user inputs after which
                current state returns to root
        """
        cue_emb = self._compute_embedding(cue_line)
        self.dialog_trees[speaker_id].create_node(
            None,
            node_id,
            parent=parent,
            data=DialogLine(cue_emb, script_line, expires_after, threshold)
        )

    def _compute_embedding(self, line: str) -> np.ndarray:
        ids = np.asarray(
            self.tokenizer.encode(line)
        ).reshape([1, -1]).astype(np.int64)
        attention_mask = np.ones_like(ids)
        outp = self.model.run(None, {
            'input_ids': ids, 'attention_mask': attention_mask
        })
        return outp[1]

    def delete_speaker(self, speaker_id: int):
        del self.dialog_trees[speaker_id]
        del self.current_nodes[speaker_id]
        del self.no_activation_cnts[speaker_id]
