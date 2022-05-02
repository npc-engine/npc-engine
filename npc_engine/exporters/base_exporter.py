"""Module with Exporters base class."""
from typing import Any, List
from abc import ABC, abstractmethod
import inspect
import click
from npc_engine.server.utils import start_test_server


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

    @classmethod
    @abstractmethod
    def get_api(cls) -> str:
        """Get the api for the exporter."""
        pass

    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Get the model name."""
        pass

    def test_model(self, models_path: str, model_id: str):
        """Test the model."""
        start_server = click.confirm(
            "Start testing server? (It should be already running otherwise)"
        )
        if start_server:
            start_test_server("5555", models_path)
        self.test_model_impl(models_path, model_id)

    @abstractmethod
    def test_model_impl(self, models_path: str, model_id: str):
        """Test the model."""
        pass
