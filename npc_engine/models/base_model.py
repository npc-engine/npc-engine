"""Module with Model base class."""
from abc import ABC
import os
import yaml


class Model(ABC):
    """Abstract base class for managed models."""

    models = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where model classes get registered to be loadable."""
        super().__init_subclass__(**kwargs)
        cls.models[cls.__name__] = cls

    @classmethod
    def load(cls, path: str):  # pragma: no cover
        """Load the model from the path."""
        config_path = os.path.join(path, "config.yml")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["model_path"] = path
        model_cls = cls.models[config_dict["model_type"]]
        return model_cls(**config_dict)
