"""Control interface client implementation."""
import zmq
from typing import Any, Dict
from npc_engine.service_clients.service_client import ServiceClient


class ControlClient(ServiceClient):
    """Json rpc client for control requests."""

    def __init__(self, zmq_context: zmq.Context, port: str):
        """Connect to the server on the port."""
        super().__init__(zmq_context, port, "control")

    def start_service_request(self, service_id):
        """Send a start service request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "start_service",
            "id": 0,
            "params": [service_id],
        }
        self.send_request(request)

    def stop_service_request(self, service_id) -> Dict[str, Any]:
        """Send a stop service request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "stop_service",
            "id": 0,
            "params": [service_id],
        }
        self.send_request(request)

    def get_service_status_request(self, service_id) -> Dict[str, Any]:
        """Send a get service status request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "get_service_status",
            "id": 0,
            "params": [service_id],
        }
        return self.send_request(request)

    def restart_service_request(self, service_id) -> Dict[str, Any]:
        """Send a restart service request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "restart_service",
            "id": 0,
            "params": [service_id],
        }
        self.send_request(request)

    def get_services_metadata_request(self) -> Dict[str, Any]:
        """Send a get services metadata request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "get_services_metadata",
            "id": 0,
            "params": [],
        }
        return self.send_request(request)
