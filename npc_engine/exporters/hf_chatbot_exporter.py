"""Exporter implementation for the Huggingface text generation models."""
import time
from npc_engine.exporters.base_hf_exporter import BaseHfExporter
from jinja2schema import infer, to_json_schema
import click
import yaml
import os
import json
import zmq

from npc_engine.service_clients import ControlClient, TextGenerationClient
from npc_engine.server.utils import schema_to_json


class HfChatbotExporter(BaseHfExporter):
    """Base exporter for Huggingface transformer models to chatbot API."""

    SUPPORTED_FEATURES = [
        "causal-lm",
        "causal-lm-with-past",
        "seq2seq-lm",
        "seq2seq-lm-with-past",
    ]

    @classmethod
    def get_api(cls) -> str:
        """Get the api for the exporter."""
        return "TextGenerationAPI"

    @classmethod
    def get_model_name(cls):
        """Get the model name."""
        return "HfChatbot"

    def create_config(self, export_path: str):
        """Create the config for the model."""
        config_dict = {}
        config_dict["model_type"] = self.get_model_name()
        config_dict["template_string"] = click.edit(
            "{#\n"
            + "Please create a template that will map context fields to prompt\n"
            + "Any context fields defined here must be then sent over the request via json as context arg\n"
            + "See Jinja docs for template design https://jinja.palletsprojects.com/en/3.1.x/templates/# \n"
            + "this example expects a list of strings named history\n"
            + "#}\n"
            + "{% for line in history %}\n"
            + "{{ line }}\n"
            + "{% endfor -%}"
        )
        config_dict["min_length"] = click.prompt(
            "Minimum generation length:", type=int, default=3
        )
        config_dict["max_length"] = click.prompt(
            "Maximum generation length (affects performance):", type=int, default=128
        )
        config_dict["repetition_penalty"] = click.prompt(
            "Repetition penalty (probability multiplier for repeated tokens):",
            type=float,
            default=1.0,
        )
        yaml.dump(config_dict, open(os.path.join(export_path, "config.yml"), "w"))

    def get_export_feature(self) -> str:
        """Select and return hf feature for export."""
        return click.prompt(
            "Please select one of the supported features (refer to huggingface docs for details):",
            type=click.Choice(self.SUPPORTED_FEATURES),
            show_default=False,
        )

    def test_model_impl(self, models_path: str, model_id: str):
        """Run test request.

        Args:
            models_path: path to models
            model_id: model id (directory name of the model)
        """
        config_path = os.path.join(models_path, model_id, "config.yml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        use_default_text = click.confirm("Use default text for variables?")
        schema = to_json_schema(infer(config["template_string"]))
        if use_default_text:

            def get_text(_):
                return "Hello world"

        else:

            def get_text(field_name):
                click.prompt(f"Please enter {field_name}:")

        context = schema_to_json(schema, get_text)
        print(f"Context: {context}")
        zmq_context = zmq.Context()
        control_client = ControlClient(zmq_context)
        chatbot_client = TextGenerationClient(zmq_context, model_id)
        control_client.start_service(model_id)
        time.sleep(1)
        response = None
        while response is None:
            try:
                response = chatbot_client.generate_reply(context)
            except RuntimeError as e:
                if "is not running" in str(e):
                    print("Model is not running, waiting for it to start..")
                    time.sleep(1)
                else:
                    raise e
        click.echo(click.style("Request context:", fg="green"))
        click.echo(json.dumps(context, indent=2))
        click.echo(click.style("Reply:", fg="green"))
        click.echo(json.dumps(response, indent=2))
