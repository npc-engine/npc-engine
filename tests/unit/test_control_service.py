"""Model manager test."""
import json
import os
import sys
import asyncio
import time
from importlib_metadata import metadata
import pytest
import zmq
import zmq.asyncio
from npc_engine.server.control_service import ControlService, ServiceState
from npc_engine.server.metadata_manager import MetadataManager


class TestControlService:
    """Test that starts npc-engine server and tests all the APIs"""

    def setup_class(cls):
        cls.context = zmq.asyncio.Context()
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path = os.path.join(
            os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1]),
            "resources",
            "models",
        )

        cls.metadata = MetadataManager(path, "5555")

    def test_service_manager_start_stop_service(self):
        """Test if models are printed without error."""

        model_manager = ControlService(self.context, self.metadata)
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.stop_service(service)
        assert model_manager.get_service_status(service) == ServiceState.STOPPED

    def test_service_manager_start_error_service(self):
        """Test if models are printed without error."""

        model_manager = ControlService(self.context, self.metadata)
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.services[service]["process"].terminate()
        time.sleep(0.5)
        with pytest.raises(ValueError):
            model_manager.get_service_status(service)
        assert model_manager.get_service_status(service) == ServiceState.ERROR

    def test_service_manager_restart_service(self):
        """Test if models are printed without error."""

        model_manager = ControlService(self.context, self.metadata)
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.restart_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING

    def test_service_manager_handle_request(self):
        """Test if models are printed without error."""

        model_manager = ControlService(self.context, self.metadata)
        address = "mock-distilgpt2"
        request = json.dumps(
            {
                "id": "HfChatbot",
                "method": "get_prompt_template",
                "params": [],
                "jsonrpc": "2.0",
            }
        )
        with pytest.raises(ValueError, match="Service mock-distilgpt2 is not running"):
            asyncio.run(model_manager.handle_request(address, request))

    def teardown_class(cls):
        cls.context.destroy()
