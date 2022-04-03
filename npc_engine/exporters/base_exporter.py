"""Module with Exporters base class."""
from typing import Any, List
from abc import ABC, abstractmethod
from click import echo
import inspect


class Exporter(ABC):
    """Abstract base class for exporter.

    Exporters are classes that handle converting models to be used with npc-engine.
    """

    exporters = {}

    def __init_subclass__(cls, **kwargs):
        """Init subclass where model classes get registered to be loadable."""
        super().__init_subclass__(**kwargs)
        cls.exporters[cls.__name__] = cls

    @classmethod
    def get_exporters(cls) -> List[Any]:
        """Create all exporters."""
        return [
            cls.exporters[name]()
            for name in cls.exporters
            if not inspect.isabstract(cls.exporters[name])
        ]

    @classmethod
    def description(cls) -> str:
        """Print the exporter."""
        return cls.__name__ + "\n\t" + cls.__doc__.split("\n\n")[0]

    @abstractmethod
    def export(self, model_path: str, export_path: str):
        """Export the model to the export path."""
        pass

    @abstractmethod
    def create_config(self, export_path: str):
        """Create the config for the model."""
        pass

    @abstractmethod
    def get_api(self) -> str:
        """Get the api for the exporter."""
        pass
