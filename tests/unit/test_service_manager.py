"""Model manager test."""
import os
import time
import pytest
from threading import Thread
import asyncio
from npc_engine.service_manager.service_manager import ServiceManager, ServiceState
import pprint


def test_model_manager_get_metadata():
    """Test if all api methods are registered"""
    path = os.path.join(
        os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1]),
        "resources",
        "models",
    )
    model_manager = ServiceManager(path)
    metadata = model_manager.get_services_metadata()
    paths = [
        f.path
        for f in os.scandir(path)
        if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
    ]
    assert len(metadata) == len(paths)
    for metadata_item in metadata:
        assert metadata_item["id"] in [path.split(os.path.sep)[-1] for path in paths]
    for metadata_item in metadata:
        assert metadata_item["path"] in paths


def test_model_manager_start_stop_service():
    """Test if models are printed without error."""
    model_manager = ServiceManager(
        os.path.join(os.path.dirname(__file__), "..", "resources", "models")
    )
    service = next(iter(model_manager.services.keys()))
    assert model_manager.get_service_status(service) == ServiceState.STOPPED
    model_manager.start_service(service)
    assert model_manager.get_service_status(service) == ServiceState.RUNNING
    model_manager.stop_service(service)
    assert model_manager.get_service_status(service) == ServiceState.STOPPED


def test_model_manager_start_error_service():
    """Test if models are printed without error."""
    model_manager = ServiceManager(
        os.path.join(os.path.dirname(__file__), "..", "resources", "models")
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


def test_model_manager_restart_service():
    """Test if models are printed without error."""
    model_manager = ServiceManager(
        os.path.join(os.path.dirname(__file__), "..", "resources", "models")
    )
    service = next(iter(model_manager.services.keys()))
    assert model_manager.get_service_status(service) == ServiceState.STOPPED
    model_manager.start_service(service)
    assert model_manager.get_service_status(service) == ServiceState.RUNNING
    model_manager.restart_service(service)
    assert model_manager.get_service_status(service) == ServiceState.RUNNING
