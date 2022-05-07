"""Module that implements ZMQ base client communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
from typing import Any, Dict
import zmq
import zmq.asyncio
from abc import ABC, abstractclassmethod

from loguru import logger

from npc_engine.server.utils import build_ipc_uri


class ServiceClient(ABC):
    """Base json rpc client."""

    clients = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where service classes get registered to be discovered."""
        super().__init_subclass__(**kwargs)
        cls.clients[cls.get_api_name()] = cls

    def __init__(self, zmq_context: zmq.Context, service_id: str = None):
        """Connect to the server on the port."""
        self.context = zmq_context
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(
            zmq.IDENTITY,
            service_id.encode("utf-8")
            if service_id
            else self.get_api_name().encode("utf-8"),
        )
        self.socket.connect(build_ipc_uri("self"))
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

    @abstractclassmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        pass

    @classmethod
    def get_api_client(cls, api_name: str) -> "ServiceClient":
        """Return the client for the api."""
        try:
            return cls.clients[api_name]
        except KeyError:
            raise RuntimeError(
                f"Client for the API {api_name} not found. Available clients: {list(cls.clients.keys())}"
            )
