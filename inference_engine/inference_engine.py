from typing import Dict, List, Any
from .text_generation import BartChatbot
from .speech_synthesis import TTS
from .semantic_tests import SemanticTests
import logging
import traceback


class InferenceEngine:
    def __init__(self, chatbot_path, tts_path, roberta_path):
        self.chatbot = BartChatbot(chatbot_path)
        self.tts = TTS(tts_path)
        self.semantic_tests = SemanticTests(roberta_path)
        self.INCORRECT_MSG = 1

    def handle_message(self, message):
        if message["cmd"] == "tts":
            return self.handle_tts(message)
        elif message["cmd"] == "add_test":
            return self.handle_add_test(message)
        elif message["cmd"] == "test":
            return self.handle_test(message)
        elif message["cmd"] == "chatbot":
            return self.handle_chatbot(message)

    def handle_tts(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ["voice_id", "line"]):
            return {"status": self.INCORRECT_MSG}
        return {
            "status": 0,
            "audio": self.tts.tts(message["voice_id"], message["line"]),
        }

    def add_test(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ["test_id", "lines"]):
            return {"status": self.INCORRECT_MSG}
        self.semantic_tests.add_test(message["test_id"], message["lines"])
        return {
            "status": 0,
        }

    def handle_test(self, message: Dict[str, Any]):
        if self._validate_msg_fields(message, ["test_ids", "line", "method"]):
            return {
                "status": 0,
                "results": self.semantic_tests.test(
                    message["line"], message["test_ids"], message["method"]
                ),
            }
        elif self._validate_msg_fields(message, ["line", "query_lines", "method"]):
            return {
                "status": 0,
                "results": self.semantic_tests.test_custom(
                    message["line"], message["query_lines"], message["method"]
                ),
            }
        else:
            return {"status": self.INCORRECT_MSG}

    def handle_chatbot(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(
            message, ["persona", "history", "temperature", "topk"]
        ):
            return {"status": self.INCORRECT_MSG}
        return {
            "status": 0,
            "reply": self.chatbot.generate_reply(
                message["persona"],
                message["history"],
                message["temperature"],
                message["topk"],
            ),
        }

    def _validate_msg_fields(self, msg: Dict[str, Any], fields: List[str]) -> bool:
        msg_correct = True
        for field in fields:
            msg_correct = msg_correct and (field in msg)
        return msg_correct
