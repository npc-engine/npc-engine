"""Module that implements ZMQ base client communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
from typing import Any, Dict
import zmq
import zmq.asyncio

from loguru import logger


class ServiceClient:
    """Base json rpc client."""

    def __init__(self, zmq_context: zmq.Context, port: str, service_id: str):
        """Connect to the server on the port."""
        self.context = zmq_context
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IDENTITY, service_id.encode("utf-8"))
        self.socket.connect(f"tcp://localhost:{port}")
        logger.info("Connected to server")

    def send_request(self, request: Dict[str, Any]) -> Any:
        """Send request to the server and return the response.

        Args:
            request: The request to send to the server.

        Returns:
            The result from the server.
        """
        logger.trace(f"Sending request: {request}")
        self.socket.send_json(request)
        response = self.socket.recv_json()
        logger.trace(f"Received response: {response}")
        if "result" in response:
            return response["result"]
        elif "code" in response:
            raise RuntimeError(f"code: {response['code']}. {response['message']}")
