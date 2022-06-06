"""Module with Model base class."""
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
import os
import zmq
import onnxruntime as rt
from loguru import logger
from jsonrpc import JSONRPCResponseManager, Dispatcher
from pathlib import Path
from npc_engine.service_clients.control_client import ControlClient
from npc_engine.services.factory_mixin import FactoryMixin


class BaseService(FactoryMixin, ABC):
    """Abstract base class for managed services."""

    def __init__(
        self,
        service_id: str,
        context: zmq.Context,
        uri: str,
        providers: List[str] = None,
        *args,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            context (zmq.Context): ZMQ context
            uri (str): URI to serve requests to
            dependency_clients (list(ServiceClient)): List of dependency clients
        """
        super(BaseService, self).__init__()
        self.zmq_context = context
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        if uri.startswith("ipc://"):
            os.makedirs(Path(uri.replace("ipc://", "")).parent, exist_ok=True)
            os.chmod(Path(uri.replace("ipc://", "")).parent, 777)
        self.socket.bind(uri)
        self.service_id = service_id
        self.control_client = None
        self._set_and_validate_providers(providers)

    @classmethod
    @abstractmethod
    def get_api_name(cls) -> str:
        """Return the name of the API."""
        pass

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

    def loop(self):
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

    def get_providers(self) -> List[str]:
        """Return onnxruntime providers to use."""
        return self.providers

    def _set_and_validate_providers(self, providers: List[str]):
        if providers is not None:
            self.providers = providers
            for provider in self.providers:
                if provider == "gpu":
                    provider = [
                        prov
                        for prov in rt.get_available_providers()
                        if "DML" in prov or "CUDA" in prov or "Tensorrt" in prov
                    ][0]
                if provider == "cpu":
                    provider = [
                        prov for prov in rt.get_available_providers() if "CPU" in prov
                    ][0]
                if provider not in rt.get_available_providers():
                    raise RuntimeError(f"Provider {provider} is not available")
        else:
            self.providers = rt.get_available_providers()
