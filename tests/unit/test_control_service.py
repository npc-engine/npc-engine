"""Model manager test."""
import json
import os
import sys
import asyncio
import time
import pytest
import zmq
import zmq.asyncio
import npc_engine.server.control_service
from npc_engine.server.control_service import ControlService, ServiceState
from npc_engine.server.metadata_manager import MetadataManager
from loguru import logger


def wrapped_service(*args, **kwargs):
    import coverage

    coverage.process_startup()
    return npc_engine.server.control_service.service_process(*args, **kwargs)


class TestControlService:
    """Test that starts npc-engine server and tests all the APIs"""

    def setup_class(cls):
        cls.context = zmq.asyncio.Context(5)
        path = os.path.join(
            os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1]),
            "resources",
            "models",
        )
        logger.remove()
        logger.add(sys.stdout, level="DEBUG", enqueue=True)
        cls.metadata = MetadataManager(path, "5555")

    @pytest.mark.asyncio
    async def test_service_manager_start_stop_service(self):
        """Test if models are printed without error."""
        os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"

        old_sp = npc_engine.server.control_service.service_process
        npc_engine.server.control_service.service_process = wrapped_service
        print("STARTING")
        model_manager = ControlService(self.context, self.metadata)
        print("STARTED")
        service = "mock-paraphrase-MiniLM-L6-v2"
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        print("GET SERVICE STATUS")
        with pytest.raises(ValueError, match=f"Service {service} is not running"):
            print("before stop_service")
            model_manager.stop_service(service)
        print("stop_service")
        model_manager.start_service(service)
        print("started")
        while model_manager.services[service]["dispatch_coroutine"] is None:
            await asyncio.sleep(0.1)
        print("dispatch_coroutine")
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.stop_service(service)
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        del model_manager
        npc_engine.server.control_service.service_process = old_sp

    @pytest.mark.asyncio
    async def test_service_manager_start_error_service(self):
        """Test if models are printed without error."""
        os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"

        old_sp = npc_engine.server.control_service.service_process
        npc_engine.server.control_service.service_process = wrapped_service

        model_manager = ControlService(self.context, self.metadata)
        service = "mock-paraphrase-MiniLM-L6-v2"
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)

        while model_manager.services[service]["dispatch_coroutine"] is None:
            await asyncio.sleep(0.1)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.services[service]["process"].kill()
        model_manager.services[service]["process"].terminate()
        with pytest.raises(ValueError):
            model_manager.get_service_status(service)
        assert model_manager.get_service_status(service) == ServiceState.ERROR
        npc_engine.server.control_service.service_process = old_sp

    @pytest.mark.asyncio
    async def test_service_manager_restart_service(self):
        """Test if models are printed without error."""
        os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"

        old_sp = npc_engine.server.control_service.service_process
        npc_engine.server.control_service.service_process = wrapped_service

        model_manager = ControlService(self.context, self.metadata)
        service = "mock-paraphrase-MiniLM-L6-v2"
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)

        while model_manager.services[service]["dispatch_coroutine"] is None:
            await asyncio.sleep(0.1)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.restart_service(service)

        while model_manager.services[service]["dispatch_coroutine"] is None:
            await asyncio.sleep(0.1)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        npc_engine.server.control_service.service_process = old_sp

    @pytest.mark.asyncio
    async def test_service_manager_handle_request(self):
        """Test if models are printed without error."""
        os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"

        old_sp = npc_engine.server.control_service.service_process
        npc_engine.server.control_service.service_process = wrapped_service

        model_manager = ControlService(self.context, self.metadata)
        model_manager.start_service("mock-distilgpt2")

        while model_manager.services["mock-distilgpt2"]["dispatch_coroutine"] is None:
            await asyncio.sleep(0.1)
        address = "mock-distilgpt2"
        request = json.dumps(
            {
                "id": "HfChatbot",
                "method": "get_prompt_template",
                "params": [],
                "jsonrpc": "2.0",
            }
        )
        assert model_manager.get_service_status(address) == ServiceState.RUNNING
        await model_manager.handle_request(address, request)
        npc_engine.server.control_service.service_process = old_sp

    @pytest.mark.asyncio
    async def test_service_manager_handle_request_error(self):
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
            await model_manager.handle_request(address, request)

    @pytest.mark.asyncio
    async def test_service_manager_check_dependency(self):
        """Test check_dependency method"""
        model_manager = ControlService(self.context, self.metadata)
        service_iter = iter(model_manager.services.keys())
        service1 = next(service_iter)
        service2 = next(service_iter)
        model_manager.check_dependency(service1, service2)
        assert service2 in self.metadata.services[service1].dependencies

    def teardown_class(cls):
        cls.context.destroy()
