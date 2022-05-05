"""Chatbot test."""
from npc_engine.services.text_generation import TextGenerationAPI
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
import mocks.zmq_mocks as zmq


ctx = '{"test": ""}'

template = """
{{ bos_token }}
{{ test }}
"""


class MockChatbotModel(TextGenerationAPI):
    def __init__(self) -> None:
        super().__init__(template, context=zmq.Context(), uri="inproc://test")

    def run(self, prompt: str, temperature: float = 1, topk: int = None):
        assert (
            prompt
            == """
{BOS_TOKEN}
test"""
        )
        return "success"

    def get_special_tokens(self):
        return {"bos_token": "{BOS_TOKEN}"}


def test_chatbot_api():

    chatbot = MockChatbotModel()

    result = chatbot.generate_reply({"test": "test"})
    assert "success" == result
