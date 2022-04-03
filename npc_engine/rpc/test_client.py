"""Module that implements ZMQ server communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
from typing import Any, Dict
import zmq
from loguru import logger


class TestClient:
    """Json rpc client for testing."""

    def __init__(self, port: str):
        """Connect to the server on the port."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        logger.info("Connected to server")

    def chatbot_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
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
        logger.info("Sending request: %s" % request)
        self.socket.send_json(request)
        logger.info("Waiting for response")
        response = self.socket.recv_json()
        logger.info("Received response: %s" % response)
        return response
