"""Module that implements management and loading of the models."""
import os

from inference_engine import models


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

    def build_api_dict(self):
        """Build api dict.

        Returns:
            dict(str,str): Mapping "method_name" -> callable
                that will be exposed to API
        """
        api_dict = {}
        for model in self.models:
            for method in type(model).API_METHODS:
                api_dict[method] = lambda *args, **kwargs: getattr(model, method)(
                    *args, **kwargs
                )
        return api_dict
