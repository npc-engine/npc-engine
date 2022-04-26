"""Huggingface chatbot interface client implementation."""
from typing import List, Tuple, Union
import zmq
from npc_engine.service_clients.service_client import ServiceClient


class SequenceClassifierClient(ServiceClient):
    """Json rpc client for chatbot service."""

    def __init__(
        self,
        zmq_context: zmq.Context,
        port: str,
        service_id: str = "SequenceClassifierAPI",
    ):
        """Connect to the server on the port."""
        super().__init__(zmq_context, port, service_id)

    def classify(self, texts: List[Union[str, Tuple[str, str]]]) -> str:
        """Send a chatbot request to the server.

        Args:
            context: A dictionary containing the chatbot request.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "classify",
            "id": 0,
            "params": [texts],
        }
        reply = self.send_request(request)
        return reply
