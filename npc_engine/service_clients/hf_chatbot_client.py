"""Huggingface chatbot interface client implementation."""
from typing import Any, Dict
import zmq
from npc_engine.service_clients.service_client import ServiceClient


class HfChatbotClient(ServiceClient):
    """Json rpc client for chatbot service."""

    def __init__(
        self, zmq_context: zmq.Context, port: str, service_id: str = "HfChatbot"
    ):
        """Connect to the server on the port."""
        super().__init__(zmq_context, port, service_id)

    def generate_reply_request(self, context: Dict[str, Any]) -> str:
        """Send a chatbot request to the server.

        Args:
            context: A dictionary containing the chatbot request.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "generate_reply",
            "id": 0,
            "params": [context],
        }
        reply = self.send_request(request)
        return reply

    def get_prompt_template_request(self) -> str:
        """Send a chatbot request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "get_prompt_template",
            "id": 0,
            "params": [],
        }
        return self.send_request(request)

    def get_context_template_request(self) -> Dict[str, Any]:
        """Send a chatbot request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "get_context_template",
            "id": 0,
            "params": [],
        }
        return self.send_request(request)

    def get_special_tokens_request(self) -> Dict[str, Any]:
        """Send a chatbot request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "get_special_tokens",
            "id": 0,
            "params": [],
        }
        return self.send_request(request)
