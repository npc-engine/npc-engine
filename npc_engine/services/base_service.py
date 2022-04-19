"""Module with Model base class."""
from typing import Dict
from abc import ABC
import os
import yaml
import zmq
from loguru import logger
from jsonrpc import JSONRPCResponseManager, Dispatcher


class BaseService(ABC):
    """Abstract base class for managed services."""

    models = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where service classes get registered to be discovered."""
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

    def __init__(self, context: zmq.Context, uri: str, *args, **kwargs):
        """Initialize the service."""
        self.zmq_context = context
        self.socket = context.socket(zmq.REP)
        self.socket.bind(uri)

    @classmethod
    def create(cls, context: zmq.Context, path: str, uri: str):
        """Create a service from the path."""
        config_path = os.path.join(path, "config.yml")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["model_path"] = path
        config_dict["uri"] = uri
        model_cls = cls.models[config_dict["model_type"]]
        return model_cls(**config_dict, context=context)

    def start(self):
        """Run service main loop that accepts json rpc over pipes."""
        dispatcher = Dispatcher()
        dispatcher.update(self.build_api_dict())
        dispatcher.update({"status": self.status})
        while True:
            request = self.socket.recv_string()
            response = JSONRPCResponseManager.handle(request, dispatcher)
            self.socket.send_string(response.json)

    def status(self):
        """Return status of the service."""
        from npc_engine.service_manager.service_manager import ServiceState

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
                f"Registering method {method} for model {type(self).__name__}"
            )  # TODO
            api_dict[method] = getattr(self, method)
        return api_dict

    @classmethod
    def get_metadata(cls, path: str) -> Dict[str, str]:
        """Print the model from the path."""
        path = path.replace("\\", os.path.sep)
        model_id = path.split(os.path.sep)[-1]
        config_path = os.path.join(path, "config.yml")
        readme_path = os.path.join(path, "README.md")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        try:
            with open(readme_path) as f:
                readme = f.read().split("---")[-1]
        except FileNotFoundError:
            readme = ""
        return {
            "id": model_id,
            "service": config_dict["model_type"],
            "path": path,
            "service_short_description": cls.models[
                config_dict["model_type"]
            ].__doc__.split("\n\n")[0],
            "service_description": cls.models[config_dict["model_type"]].__doc__,
            "readme": readme,
        }
