from typing import Dict, List, Any
from .semantic_tests import SemanticTests
from inference_engine import models


class InferenceEngine:
    def __init__(self, chatbot_path, tts_path, roberta_path):
        print(f"models {models.Model.models}")
        self.chatbot = models.Model.load(chatbot_path)
        self.tts = models.Model.load(tts_path)
        self.semantic_tests = SemanticTests(roberta_path)
        self._tts_generator = None
        self.STATUS_OK = "OK"
        self.INCORRECT_MSG = "Incorrect message received"
        self.TTS_NOT_STARTED = "TextToSpeech: Tried to get next speech chunk but no speech generation was started"

    def handle_message(self, message):
        if message["cmd"] == "start_tts":
            return self.handle_tts(message)
        elif message["cmd"] == "tts_next":
            return self.handle_tts_next(message)
        elif message["cmd"] == "add_test":
            return self.handle_add_test(message)
        elif message["cmd"] == "test":
            return self.handle_test(message)
        elif message["cmd"] == "chatbot":
            return self.handle_chatbot(message)
        elif message["cmd"] == "status":
            return {"status": self.STATUS_OK}
        else:
            return {"status": self.INCORRECT_MSG}

    def handle_tts(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ["voice_id", "line"]):
            return {"status": self.INCORRECT_MSG}
        self._tts_generator = self.tts.run(
            message["voice_id"], message["line"], message.get("n_chunks", 10)
        )
        return {
            "status": self.STATUS_OK,
        }

    def handle_tts_next(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, []):
            return {"status": self.INCORRECT_MSG}
        if self._tts_generator is None:
            return {"status": self.TTS_NOT_STARTED}
        return {
            "status": self.STATUS_OK,
            "audio": next(self._tts_generator).tolist(),
        }

    def handle_add_test(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ["test_id", "lines"]):
            return {"status": self.INCORRECT_MSG}
        self.semantic_tests.add_test(message["test_id"], message["lines"])
        return {
            "status": self.STATUS_OK,
        }

    def handle_test(self, message: Dict[str, Any]):
        if self._validate_msg_fields(message, ["test_ids", "line", "method"]):
            return {
                "status": self.STATUS_OK,
                "results": self.semantic_tests.test(
                    message["line"], message["test_ids"], message["method"]
                ),
            }
        elif self._validate_msg_fields(message, ["line", "query_lines", "method"]):
            return {
                "status": self.STATUS_OK,
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
            "status": self.STATUS_OK,
            "reply": self.chatbot.generate_reply(
                message, message["temperature"], message["topk"],
            ),
        }

    def _validate_msg_fields(self, msg: Dict[str, Any], fields: List[str]) -> bool:
        msg_correct = True
        for field in fields:
            msg_correct = msg_correct and (field in msg)
        return msg_correct
