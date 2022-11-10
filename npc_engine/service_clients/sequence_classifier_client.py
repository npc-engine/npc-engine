"""Huggingface sequence classifier interface client implementation."""
from typing import List, Tuple, Union
import zmq
from npc_engine.service_clients.service_client import ServiceClient
from npc_engine.utils import ServerRequest


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
        request = ServerRequest(jsonrpc="2.0", method="classify", id=0, params=[texts]).to_json()
        return self.send_request(request)

    @classmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        return "SequenceClassifierAPI"
