"""Exporter for huggingface sequence classification models."""
import os
import time
import zmq
import json
import yaml
import click
from npc_engine.exporters.base_hf_exporter import BaseHfExporter
from npc_engine.service_clients import ControlClient, SequenceClassifierClient


class HfClassifierExporter(BaseHfExporter):
    """Exporter for the Huggingface classification models."""

    def get_api(self) -> str:
        """Get the api for the exporter."""
        return "SequenceClassifierAPI"

    @classmethod
    def get_model_name(cls):
        """Get the model name."""
        return "HfClassifier"

    def create_config(self, export_path: str):
        """Create the config for the model."""
        config_dict = {}
        config_dict["type"] = self.get_model_name()
        config_dict["cache_size"] = click.prompt("Cache size", type=int)
        with open(os.path.join(export_path, "config.yml"), "w") as f:
            yaml.dump(config_dict, f)

    def get_export_feature(self) -> str:
        """Create the config for the model."""
        return "sequence-classification"

    def test_model_impl(self, models_path: str, model_id: str):
        """Run test request.

        Args:
            models_path: path to models
            model_id: model id (directory name of the model)
        """
        use_default_text = click.confirm("Use default text for variables?")
        if use_default_text:
            texts = [["Hello", "world"], "Hello world"]
        else:
            texts = [click.prompt("Text 1"), click.prompt("Text 2")]

        zmq_context = zmq.Context()
        control_client = ControlClient(zmq_context)
        service_client = SequenceClassifierClient(zmq_context, model_id)
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
        scores = service_client.classify(texts)

        click.echo(click.style("Scores:", fg="green"))
        click.echo(json.dumps(scores, indent=2))
        click.echo(click.style("Argmax classes:", fg="green"))
        click.echo(
            json.dumps([classes.index(max(classes)) for classes in scores], indent=2)
        )
