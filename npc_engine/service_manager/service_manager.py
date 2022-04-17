"""Module that implements lifetime and discoverability of the services."""
import asyncio
import json
from multiprocessing import Pipe, Process
import os
from typing import Dict

from npc_engine import services
from jsonrpc import JSONRPCResponseManager, Dispatcher
from collections import namedtuple
import ntpath
import yaml
from loguru import logger


class ServiceState:
    """Enum for the state of the service."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    AWAITING = "awaiting"
    TIMEOUT = "timeout"
    ERROR = "error"


ServiceDescriptor = namedtuple(
    "ServiceDescriptor", ["id", "type", "path", "process_data"]
)


def service_process(service_path: str, pipe: Pipe) -> None:
    """Service subprocess function.

    Starts the service and runs it's loop.
    """
    logger.remove()
    logger.add(
        os.path.join("logs", f"{service_path.split(os.path.sep)[-1]}.log"),
        rotation="10 MB",
        enqueue=True,
    )
    service = services.BaseService.create(service_path, pipe)
    service.start()


class ServiceManager:
    """Object for managing lifetime and discoverability of the services."""

    def __init__(self, path):
        """Create model manager and load models from the given path."""
        self.services = self._scan_path(path)
        self.control_dispatcher = Dispatcher()
        self.control_dispatcher.update(
            {
                "get_services_metadata": self.get_services_metadata,
                "get_service_status": self.get_service_status,
                "start_service": self.start_service,
                "stop_service": self.stop_service,
                "restart_service": self.restart_service,
            }
        )

    def handle_request(self, address: str, request: str) -> str:  # Add timeouts
        """Parse request string and route request to correct service.

        Args:
            address (str): address of the service (either model name or class name)
            request (str): jsonRPC string

        Returns:
            str: jsonRPC response
        """
        service_id = self.resolve_and_check_service(address)
        logger.info(f"Request from {address}\n Request: {request}")
        if service_id == "control":
            return JSONRPCResponseManager.handle(request, self.control_dispatcher).json
        else:
            if (
                self.services[service_id].process_data["state"] != ServiceState.RUNNING
                and self.services[service_id].process_data["state"]
                != ServiceState.STARTING
            ):
                raise ValueError(f"Service {service_id} is not running")
            else:
                service_pipe = self.services[service_id].process_data["pipe"]
                service_pipe.send(request)
                response = (
                    service_pipe.recv()
                )  # TODO: Check if error happens and raise exception
        return response

    def resolve_and_check_service(self, id_or_type):
        """Resolve service id or type to service id."""
        service_id = None
        if id_or_type == "control":
            service_id = "control"
        else:
            if id_or_type in self.services:
                service_id = id_or_type
            else:
                for service_key, service in self.services.items():
                    if service.type == id_or_type:
                        service_id = service_key
                        break
            if service_id is None:
                raise ValueError(f"Service {id_or_type} not found")
            if (
                self.services[service_id].process_data["state"] == ServiceState.RUNNING
                or self.services[service_id].process_data["state"]
                == ServiceState.STARTING
                or self.services[service_id].process_data["state"]
                == ServiceState.AWAITING
            ) and not self.services[service_id].process_data["process"].is_alive():
                self.services[service_id].process_data["state"] = ServiceState.ERROR
                raise ValueError(
                    f"Error in service {service_id}. Process is not alive."
                )

        return service_id

    def get_services_metadata(self):
        """List the models in the folder."""
        return [
            services.BaseService.get_metadata(descriptor.path)
            for descriptor in self.services.values()
        ]

    def get_service_status(self, service_id):
        """Get the status of the service."""
        service_id = self.resolve_and_check_service(service_id)
        return self.services[service_id].process_data["state"]

    def start_service(self, service_id):
        """Start the service."""
        service_id = self.resolve_and_check_service(service_id)
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not found")
        if self.services[service_id].process_data["state"] == ServiceState.RUNNING:
            raise ValueError(f"Service {service_id} is already running")
        pipe_main, pipe_child = Pipe()

        process = Process(
            target=service_process,
            args=(self.services[service_id].path, pipe_child),
            daemon=True,
        )
        process.start()
        self.services[service_id].process_data["process"] = process
        self.services[service_id].process_data["state"] = ServiceState.STARTING
        self.services[service_id].process_data["pipe"] = pipe_main
        try:
            asyncio.create_task(self.confirm_state_coroutine(service_id))
        except RuntimeError:
            asyncio.run(self.confirm_state_coroutine(service_id))

    async def confirm_state_coroutine(self, service_id):
        """Confirm the state of the service."""
        response = self.handle_request(
            service_id, json.dumps({"jsonrpc": "2.0", "method": "status", "id": 1})
        )
        resp_dict = json.loads(response)
        if resp_dict["result"] == ServiceState.RUNNING:
            self.services[service_id].process_data["state"] = ServiceState.RUNNING
        elif resp_dict["result"] == ServiceState.STARTING:
            logger.info(f"Service {service_id} responds but still starting")
            await asyncio.sleep(1)
            await self.confirm_state_coroutine(service_id)
        else:
            logger.warning(
                f"Service {service_id} failed to start and returned incorrect state."
            )
            self.services[service_id].process_data["state"] = ServiceState.ERROR

    def stop_service(self, service_id):
        """Stop the service."""
        service_id = self.resolve_and_check_service(service_id)
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not found")
        if self.services[service_id].process_data["state"] != ServiceState.RUNNING:
            raise ValueError(f"Service {service_id} is not running")
        self.services[service_id].process_data["process"].terminate()
        self.services[service_id].process_data["process"] = None
        self.services[service_id].process_data["state"] = ServiceState.STOPPED
        self.services[service_id].process_data["pipe"] = None

    def restart_service(self, service_id):
        """Restart the service."""
        self.stop_service(service_id)
        self.start_service(service_id)

    def _scan_path(self, path: str) -> Dict[str, ServiceDescriptor]:
        """Scan services defined in the given path."""
        norm_path = ntpath.normpath(path).replace("\\", os.path.sep)
        paths = [
            f.path
            for f in os.scandir(norm_path)
            if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
        ]
        services = {}
        for path in paths:
            with open(os.path.join(path, "config.yml")) as f:
                config_dict = yaml.safe_load(f)
                services[os.path.basename(path)] = ServiceDescriptor(
                    os.path.basename(path),
                    config_dict.get("model_type", config_dict.get("type", None)),
                    path,
                    {"process": None, "pipe": None, "state": ServiceState.STOPPED},
                )

        return services
