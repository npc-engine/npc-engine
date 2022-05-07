"""Huggingface sequence classifier interface client implementation."""
from typing import List, Tuple, Union
import zmq
from npc_engine.service_clients.service_client import ServiceClient


class SequenceClassifierClient(ServiceClient):
    """Json rpc client for sequence classifier service."""

    def __init__(
        self,
        zmq_context: zmq.Context,
        service_id: str = "SequenceClassifierAPI",
    ):
        """Connect to the server on the port."""
        super().__init__(zmq_context, service_id)

    def classify(self, texts: List[Union[str, Tuple[str, str]]]) -> str:
        """Send a classify request to the SequenceClassifierAPI.

        Args:
            texts: Batch of texts to classify.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "classify",
            "id": 0,
            "params": [texts],
        }
        reply = self.send_request(request)
        return reply

    @classmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        return "SequenceClassifierAPI"
