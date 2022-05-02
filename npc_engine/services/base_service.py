"""Module with Model base class."""
from abc import ABC, abstractmethod
import os
from typing import List
import yaml
import zmq
from loguru import logger
from jsonrpc import JSONRPCResponseManager, Dispatcher
from pathlib import Path

from npc_engine.services.utils.config import get_type_from_dict
from npc_engine.service_clients import ServiceClient


class BaseService(ABC):
    """Abstract base class for managed services."""

    # A list of resolvable names of services that Service depends on
    DEPENDENCIES = []

    models = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where service classes get registered to be discovered."""
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

    def __init__(
        self,
        context: zmq.Context,
        uri: str,
        dependency_clients: List[ServiceClient] = [],
        *args,
        **kwargs,
    ):
        """Initialize the service."""
        self.zmq_context = context
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        if uri.startswith("ipc://"):
            os.makedirs(Path(uri.replace("ipc://", "")).parent, exist_ok=True)
            os.chmod(Path(uri.replace("ipc://", "")).parent, 777)
        self.socket.bind(uri)

        self.dependency_clients = dependency_clients

    @classmethod
    @abstractmethod
    def get_api_name(cls):
        """Return the name of the API."""
        pass

    @classmethod
    def create(
        cls,
        context: zmq.Context,
        path: str,
        uri: str,
        dependency_clients: List[ServiceClient] = [],
    ):
        """Create a service from the path."""
        config_path = os.path.join(path, "config.yml")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["model_path"] = path
        model_cls = cls.models[get_type_from_dict(config_dict)]
        return model_cls(
            **config_dict,
            context=context,
            uri=uri,
            dependency_clients=dependency_clients,
        )

    def get_client(self, name: str):
        """Get a dependency client by name to use it in service logic."""
        try:
            return self.dependency_clients[self.DEPENDENCIES.index(name)]
        except ValueError:
            raise ValueError(
                f"No dependency client with name {name}."
                + "Please make sure you specified all dependencies in the `DEPENDENCIES` class variable."
            )

    def start(self):
        """Run service main loop that accepts json rpc."""
        try:
            dispatcher = Dispatcher()
            dispatcher.update(self.build_api_dict())
            dispatcher.update({"status": self.status})
            while True:
                request = self.socket.recv_string()
                response = JSONRPCResponseManager.handle(request, dispatcher)
                self.socket.send_string(response.json)
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            self.socket.close()
            self.zmq_context.destroy()

    def status(self):
        """Return status of the service."""
        from npc_engine.server.control_service import ServiceState

        return ServiceState.RUNNING

    def build_api_dict(self):
        """Build api dict.

        Returns:
            dict(str,str): Mapping "method_name" -> callable
                that will be exposed to API
        """
        api_dict = {}
        for method in type(self).API_METHODS:
            logger.info(
                f"Registering method {method} for service {type(self).__name__}"
            )
            api_dict[method] = getattr(self, method)
        return api_dict
