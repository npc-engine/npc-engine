"""Factory mixin for service."""
import os
import yaml
import zmq
from npc_engine.services.utils.config import get_type_from_dict


class FactoryMixin:
    """Mixin base class for services that can be created via create method."""

    models = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where service classes get registered to be discovered."""
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

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
