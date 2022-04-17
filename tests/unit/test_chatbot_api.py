"""Chatbot test."""
from npc_engine.services.chatbot import ChatbotAPI

ctx = '{"test": ""}'

template = """
{{ bos_token }}
{{ test }}
"""


class MockChatbotModel(ChatbotAPI):
    def __init__(self) -> None:
        super().__init__(template, ctx)

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
