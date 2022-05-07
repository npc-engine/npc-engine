"""Module with Model base class."""
from abc import ABC, abstractmethod
import os
from typing import Callable, Dict
import yaml
import zmq
from loguru import logger
from jsonrpc import JSONRPCResponseManager, Dispatcher
from pathlib import Path
from npc_engine.service_clients.control_client import ControlClient

from npc_engine.services.utils.config import get_type_from_dict


class BaseService(ABC):
    """Abstract base class for managed services."""

    models = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where service classes get registered to be discovered."""
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

    def __init__(
        self,
        service_id: str,
        context: zmq.Context,
        uri: str,
        *args,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            context (zmq.Context): ZMQ context
            uri (str): URI to serve requests to
            dependency_clients (list(ServiceClient)): List of dependency clients
        """
        self.zmq_context = context
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        if uri.startswith("ipc://"):
            os.makedirs(Path(uri.replace("ipc://", "")).parent, exist_ok=True)
            os.chmod(Path(uri.replace("ipc://", "")).parent, 777)
        self.socket.bind(uri)
        self.service_id = service_id
        self.control_client = None

    @classmethod
    @abstractmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        pass

    @classmethod
    def create(cls, context: zmq.Context, path: str, uri: str, service_id: str):
        """Create a service from the path.

        Args:
            context (zmq.Context): ZMQ context
            path (str): Path to the service
            uri (str): URI to serve requests to

        Returns:
            Service: Service instance
        """
        config_path = os.path.join(path, "config.yml")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["model_path"] = path
        model_cls = cls.models[get_type_from_dict(config_dict)]
        return model_cls(**config_dict, context=context, uri=uri, service_id=service_id)

    def create_client(self, name: str):
        """Get a dependency client by name to use it in service logic.

        Args:
            name (str): Name of the dependency

        Returns:
            ServiceClient: Client for the dependency
        """
        if self.control_client is None:
            self.control_client = ControlClient(self.zmq_context)
        if name == "control":
            return self.control_client
        self.control_client.check_dependency(self.service_id, name)
        api_name = self.control_client.get_service_metadata(name)["api_name"]
        return ControlClient.get_api_client(api_name)(self.zmq_context, name)

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
        """Return status of the service.

        Returns:
            ServiceState
        """
        from npc_engine.server.control_service import ServiceState

        return ServiceState.RUNNING

    def build_api_dict(self) -> Dict[str, Callable]:
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
