"""Module that implements management and loading of the models."""
import os

from npc_engine import models
from loguru import logger


class ModelManager:
    """Loads the models and creates global API dictionary."""

    def __init__(self, path):
        """Create model manager and load models from the given path."""
        subdirs = [
            f.path
            for f in os.scandir(path)
            if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
        ]
        self.models = [models.Model.load(subdir) for subdir in subdirs]

    def _build_api_call(self, model, method):
        method_callable = getattr(model, method)

        def api_call(*args, **kwargs):
            logger.debug(
                f"Api call to {method} from model {type(model).__name__} "
                + f"with *args = {args} **kwargs = {kwargs}",
            )
            return method_callable(*args, **kwargs)

        return api_call

    def build_api_dict(self):
        """Build api dict.

        Returns:
            dict(str,str): Mapping "method_name" -> callable
                that will be exposed to API
        """
        api_dict = {}
        for model in self.models:
            for method in type(model).API_METHODS:
                logger.info(
                    f"Registering method {method} for model {type(model).__name__}"
                )
                api_dict[method] = self._build_api_call(model, method)
        return api_dict
