"""Exporter implementation for the Huggingface semantic similarity models."""
import os
import yaml
import time
import zmq
import json
import click
from npc_engine.service_clients import ControlClient, SimilarityClient
from npc_engine.exporters.base_hf_exporter import BaseHfExporter


class HfSimilarityExporter(BaseHfExporter):
    """Exporter for the Huggingface transformer models."""

    def get_api(self) -> str:
        """Get the api for the exporter."""
        return "SimilarityAPI"

    @classmethod
    def get_model_name(cls):
        """Get the model name."""
        return "TransformerSemanticSimilarity"

    def create_config(self, export_path: str):
        """Create the config for the model."""
        config_dict = {}
        config_dict["type"] = self.get_model_name(export_path)
        config_dict["cache_size"] = click.prompt("Cache size", type=int)
        with open(os.path.join(export_path, "config.yml"), "w") as f:
            yaml.dump(config_dict, f)

    def get_export_feature(self) -> str:
        """Create the config for the model."""
        return "default"

    def test_model_impl(self, models_path: str, model_id: str):
        """Run test request.

        Args:
            models_path: path to models
            model_id: model id (directory name of the model)
        """
        use_default_text = click.confirm("Use default text for variables?")
        if use_default_text:
            query = "Hello world"
            context = ["Whats up world"]
        else:
            query = click.prompt("Query")
            context = [click.prompt("Context")]

        zmq_context = zmq.Context()
        control_client = ControlClient(zmq_context)
        service_client = SimilarityClient(zmq_context, model_id)
        control_client.start_service(model_id)
        ready = False
        while not ready:
            time.sleep(1)
            status = control_client.get_service_status(model_id)
            ready = status == "running"
            if status == "error":
                raise Exception("Error while loading model")
            if not ready:
                print("Waiting for service to start...")
        scores = service_client.compare(query, context)

        click.echo(click.style("Scores:", fg="green"))
        click.echo(json.dumps(scores, indent=2))
        click.echo(click.style("Argmax classes:", fg="green"))
        click.echo(
            json.dumps([classes.index(max(classes)) for classes in scores], indent=2)
        )
