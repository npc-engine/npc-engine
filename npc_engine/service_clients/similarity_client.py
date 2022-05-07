"""Huggingface semantic similarity interface client implementation."""
from typing import List
import zmq
from npc_engine.service_clients.service_client import ServiceClient


class SimilarityClient(ServiceClient):
    """Json rpc client for semantic similarity service."""

    def __init__(self, zmq_context: zmq.Context, service_id: str = "SimilarityAPI"):
        """Connect to the server on the port."""
        super().__init__(zmq_context, service_id)

    def compare(self, query: str, context: List[str]) -> float:
        """Send a comparison request to the server.

        Args:
            query: A string to compute similarity with contexts.
            context: A list of strings to compute similiarity with query.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "compare",
            "id": 0,
            "params": [query, context],
        }
        reply = self.send_request(request)
        return reply

    @classmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        return "SimilarityAPI"
