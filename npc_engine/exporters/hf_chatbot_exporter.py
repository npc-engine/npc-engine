from npc_engine.exporters.base_hf_exporter import BaseHfExporter
import click
import yaml


class HfChatbotExporter(BaseHfExporter):
    """Base exporter for Huggingface transformer models to chatbot API."""

    SUPPORTED_FEATURES = [
        "causal-lm",
        "causal-lm-with-past",
        "seq2seq-lm",
        "seq2seq-lm-with-past",
    ]

    def get_api(self) -> str:
        """Get the api for the exporter."""
        return "ChatbotAPI"

    def get_model_name(self, model_path: str):
        """Get the model name."""
        return "HfChatbot"

    def create_config(self, export_path: str):
        """Create the config for the model."""
        config_dict = {}
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
        yaml.dump(config_dict, open(export_path + "/" + "config.yml", "w"))

    def get_export_feature(self) -> str:
        """Select and return hf feature for export."""
        return click.prompt(
            "Please select one of the supported features (refer to huggingface docs for details):",
            type=click.Choice(self.SUPPORTED_FEATURES),
            show_default=False,
        )
