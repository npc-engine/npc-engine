"""Model manager test."""
import json
import os
import sys
import asyncio
import time
import pytest
import zmq
import zmq.asyncio
from npc_engine.service_manager.service_manager import ServiceManager, ServiceState


class TestServiceManager:
    """Test that starts npc-engine server and tests all the APIs"""

    def setup_class(cls):
        cls.context = zmq.asyncio.Context()
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def test_service_manager_get_metadata(self):
        """Test if all api methods are registered"""
        path = os.path.join(
            os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1]),
            "resources",
            "models",
        )

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        model_manager = ServiceManager(type(self).context, path)
        metadata = model_manager.get_services_metadata()
        paths = [
            f.path
            for f in os.scandir(path)
            if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
        ]
        assert len(metadata) == len(paths)
        for metadata_item in metadata:
            assert metadata_item["id"] in [
                path.split(os.path.sep)[-1] for path in paths
            ]
        for metadata_item in metadata:
            assert metadata_item["path"] in paths

    def test_service_manager_start_stop_service(self):
        """Test if models are printed without error."""

        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.stop_service(service)
        assert model_manager.get_service_status(service) == ServiceState.STOPPED

    def test_service_manager_start_error_service(self):
        """Test if models are printed without error."""

        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.services[service].process_data["process"].terminate()
        time.sleep(0.5)
        with pytest.raises(ValueError):
            model_manager.get_service_status(service)
        assert model_manager.get_service_status(service) == ServiceState.ERROR

    def test_service_manager_restart_service(self):
        """Test if models are printed without error."""

        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        service = next(iter(model_manager.services.keys()))
        assert model_manager.get_service_status(service) == ServiceState.STOPPED
        model_manager.start_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING
        model_manager.restart_service(service)
        assert model_manager.get_service_status(service) == ServiceState.RUNNING

    def test_service_manager_handle_request(self):
        """Test if models are printed without error."""

        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
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

    def test_service_manager_dependencies(self):
        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        model_manager.services["mock-distilgpt2"] = model_manager.services[
            "mock-distilgpt2"
        ]._replace(dependencies=["SimilarityAPI"])
        model_manager.check_dependencies()
        model_manager.check_dependency_cycles()

    def test_service_manager_no_dep(self):
        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        model_manager.services["mock-distilgpt2"] = model_manager.services[
            "mock-distilgpt2"
        ]._replace(dependencies=["123"])
        print(model_manager.services["mock-distilgpt2"].id)
        with pytest.raises(
            ValueError,
            match="Service mock-distilgpt2 requires 123 service to be present.",
        ):
            model_manager.check_dependencies()

    def test_service_manager_dep_cycle(self):
        model_manager = ServiceManager(
            type(self).context,
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"),
        )
        model_manager.services["mock-distilgpt2"] = model_manager.services[
            "mock-distilgpt2"
        ]._replace(dependencies=["SimilarityAPI"])
        model_manager.services["mock-paraphrase-MiniLM-L6-v2"].dependencies.append(
            "ChatbotAPI"
        )
        with pytest.raises(ValueError, match="There are dependency cycles"):
            model_manager.check_dependency_cycles()

    def teardown_class(cls):
        cls.context.destroy()
