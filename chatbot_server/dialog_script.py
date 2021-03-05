from typing import Union, Dict, Tuple, List
from collections import namedtuple
import numpy as np
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from chatbot_server.tokenization.tokenization_roberta import RobertaTokenizerFast
import os
from treelib import Tree
from scipy.spatial.distance import cosine
import random

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
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path, fast=True)
        self.dialog_trees: Dict[str, Tree] = {}
        self.current_nodes: Dict[str, List[str]] = {}
        self.visited_nodes: Dict[str, List[str]] = {}
        self.no_activation_cnts: Dict[str, List[int]] = {}

    def add_speaker(self, speaker_id: str):
        self.dialog_trees[speaker_id] = Tree()
        self.dialog_trees[speaker_id].create_node("Root node", "root")
        self.current_nodes[speaker_id] = ["root"]
        self.visited_nodes[speaker_id] = []
        self.no_activation_cnts[speaker_id] = [0]

    def reset_state(self, speaker_id: str):
        self.current_nodes[speaker_id] = ["root"]
        self.no_activation_cnts[speaker_id] = [0]

    def step_dialog(self, speaker_id: str, line: str) -> Union[None, Tuple[str, str]]:
        reply = self._get_reply(speaker_id, line)
        self._update_inactivity_(speaker_id, reply)
        self._filter_nodes_(speaker_id)
        print("Script reply ", reply)
        return reply

    def _update_inactivity_(self, speaker_id, reply):
        if reply is None:
            self.no_activation_cnts[speaker_id] = [el + 1 for el in self.no_activation_cnts[speaker_id]]

    def _filter_nodes_(self, speaker_id):
        for node_arr_id, node_id in enumerate(self.current_nodes[speaker_id]):
            node = self.dialog_trees[speaker_id].get_node(node_id)
            if node_id != 'root':
                if self.no_activation_cnts[speaker_id][node_arr_id] >= node.data.expires_after:
                    del self.current_nodes[speaker_id][node_arr_id]
                    del self.no_activation_cnts[speaker_id][node_arr_id]
                elif node.is_leaf():
                    del self.current_nodes[speaker_id][node_arr_id]
                    del self.no_activation_cnts[speaker_id][node_arr_id]
                if node.bpointer != 'root' and all([
                    child in self.visited_nodes[speaker_id]
                    for child in
                    self.dialog_trees[speaker_id].get_node(node.bpointer).fpointer
                ]):  # If all children of the parent are visited, drop the parent
                    parent_id = self.current_nodes[speaker_id].index(node.bpointer)
                    del self.current_nodes[speaker_id][parent_id]
                    del self.no_activation_cnts[speaker_id][parent_id]

    def _get_reply(self, speaker_id: str, line: str) -> Union[None, Tuple[str, str]]:
        child_nodes = [
            child for node in self.current_nodes[speaker_id]
            for child in self.dialog_trees[speaker_id].children(node)
            if child not in self.current_nodes[speaker_id]
        ]
        if not child_nodes:
            return None
        line_emb = self._compute_embedding(line)
        similarities = [
            max([1 - cosine(line_emb, cue_emb) for cue_emb in child_node.data.cue_emb])
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
            print("script return None")
            return None
        else:
            self.current_nodes[speaker_id] += [child_nodes[reply_id].identifier]
            self.visited_nodes[speaker_id] += [child_nodes[reply_id].identifier]
            self.no_activation_cnts[speaker_id] += [0]
            print("script return smth")
            return (
                random.choice(child_nodes[reply_id].data.response),
                child_nodes[reply_id].identifier
            )

    def script_line(
        self,
        speaker_id: str,
        parent: str,
        node_id: str,
        cue_lines: List[str],
        script_lines: List[str],
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
        cue_embs = [self._compute_embedding(cue_line) for cue_line in cue_lines]
        self.dialog_trees[speaker_id].create_node(
            None,
            node_id,
            parent=parent,
            data=DialogLine(cue_embs, script_lines, expires_after, threshold)
        )

    def _compute_embedding(self, line: str) -> np.ndarray:
        ids = np.asarray(
            self.tokenizer.encode(line)
        ).reshape([1, -1]).astype(np.int64)
        print("String:", line, " IDs:", ids)
        attention_mask = np.ones_like(ids)
        outp = self.model.run(None, {
            'input_ids': ids, 'attention_mask': attention_mask
        })
        return outp[1]

    def delete_speaker(self, speaker_id: int):
        del self.dialog_trees[speaker_id]
        del self.current_nodes[speaker_id]
        del self.no_activation_cnts[speaker_id]
