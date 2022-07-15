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
{%- for line in history %}
{{ line }}
{%- endfor %}
"""


class MockChatbotModel(TextGenerationAPI):
    def __init__(self) -> None:
        super().__init__(
            context_template="",
            history_template=template,
            service_id="test",
            context=zmq.Context(),
            uri="inproc://test",
        )

    def run(self, prompt: str, temperature: float = 1, topk: int = None):
        assert (
            prompt
            == """
{BOS_TOKEN}
test
test"""
        )
        return "success"

    def get_special_tokens(self):
        return {"bos_token": "{BOS_TOKEN}"}

    def string_too_long(self, prompt: str) -> bool:
        return False


class MockChatbotModelOverflow(TextGenerationAPI):
    def __init__(self) -> None:
        super().__init__(
            context_template="",
            history_template=template,
            service_id="test",
            context=zmq.Context(),
            uri="inproc://test",
        )

    def run(self, prompt: str, temperature: float = 1, topk: int = None):
        assert (
            prompt
            == """
{BOS_TOKEN}"""
        )
        return "success"

    def get_special_tokens(self):
        return {"bos_token": "{BOS_TOKEN}"}

    def string_too_long(self, prompt: str) -> bool:
        return True


def test_chatbot_api():

    chatbot = MockChatbotModel()

    result = chatbot.generate_reply({"history": ["test", "test"]})
    assert "success" == result


def test_overflow():

    chatbot = MockChatbotModelOverflow()

    result = chatbot.generate_reply({"history": ["test", "test"]})
    assert "success" == result


def test_get_template():
    chatbot = MockChatbotModel()
    assert chatbot.get_prompt_template() == template


def test_get_context_template():
    chatbot = MockChatbotModel()
    assert chatbot.get_context_template() == {"history": [""], "bos_token": ""}
