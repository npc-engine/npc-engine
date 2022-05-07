"""Model manager test."""
import os
import pytest
import zmq
import zmq.asyncio
from npc_engine.server.metadata_manager import MetadataManager


class TestMetadataManager:
    """Test that starts npc-engine server and tests all the APIs"""

    def test_get_metadata(self):
        """Test if models are read correctly."""
        path = os.path.join(
            os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1]),
            "resources",
            "models",
        )

        model_manager = MetadataManager(path, "5555")
        metadata = model_manager.get_services_metadata()
        paths = [
            f.path
            for f in os.scandir(path)
            if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
        ]
        assert len(metadata) == len(paths)
        for metadata_item in metadata:
            assert metadata_item["id"] in [
                path.split(os.path.sep)[-1] for path in paths
            ]
        for metadata_item in metadata:
            assert metadata_item["path"] in paths

    def test_dependencies(self):
        model_manager = MetadataManager(
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"), "5555"
        )
        model_manager.services["mock-distilgpt2"] = model_manager.services[
            "mock-distilgpt2"
        ]._replace(dependencies=["SimilarityAPI"])
        model_manager.check_dependency_cycles()

    def test_dep_cycle(self):
        model_manager = MetadataManager(
            os.path.join(os.path.dirname(__file__), "..", "resources", "models"), "5555"
        )
        model_manager.services["mock-distilgpt2"] = model_manager.services[
            "mock-distilgpt2"
        ]._replace(dependencies=["SimilarityAPI"])
        model_manager.services["mock-paraphrase-MiniLM-L6-v2"] = model_manager.services[
            "mock-paraphrase-MiniLM-L6-v2"
        ]._replace(dependencies=["mock-distilgpt2"])
        with pytest.raises(ValueError, match="There are dependency cycles"):
            model_manager.check_dependency_cycles()
