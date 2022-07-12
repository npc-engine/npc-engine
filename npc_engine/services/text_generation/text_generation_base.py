"""Module that implements text generation model API."""
from typing import Dict, Any, List

from abc import abstractmethod
from npc_engine.services.base_service import BaseService
from jinja2 import Template
from jinja2schema import infer, to_json_schema

from npc_engine.server.utils import schema_to_json


class TextGenerationAPI(BaseService):
    """Abstract base class for text generation models."""

    API_METHODS: List[str] = [
        "generate_reply",
        "get_prompt_template",
        "get_special_tokens",
        "get_context_template",
    ]

    def __init__(self, context_template: str, history_template: str, *args, **kwargs):
        """Initialize prompt formatting variables.

        Args:
            template_string: Template string to be rendered as prompt.
        """
        super().__init__(*args, **kwargs)
        self.context_template_string = context_template
        self.history_template_string = history_template
        self.context_template = Template(context_template)
        self.history_template = Template(history_template)
        self.initialized = True

    @classmethod
    def get_api_name(cls) -> str:
        """Get the API name."""
        return "TextGenerationAPI"

    def generate_reply(self, context: Dict[str, Any], *args, **kwargs) -> str:
        """Format the model prompt and generate response.

        Args:
            context: Prompt context.
            *args
            **kwargs

        Returns:
            Text response to a prompt.
        """
        if not self.initialized:
            raise AssertionError(
                "Can not generate replies before Base Service class was initialized"
            )
        history = context.get("history", [])
        history_prompt = self.history_template.render(**context, **self.get_special_tokens())
        context_prompt = self.context_template.render(
            **context, **self.get_special_tokens()
        )
        prompt = context_prompt + "".join(history_prompt)
        while self.string_too_long(prompt):
            history.pop(0)
            history_prompt = self.history_template.render(
                **context, **self.get_special_tokens()
            )
            prompt = context_prompt + history_prompt
        return self.run(prompt, *args, **kwargs)

    def get_prompt_template(self) -> str:
        """Return prompt template string used to render model prompt.

        Returns:
            A template string.
        """
        return self.context_template_string + self.history_template_string

    def get_context_template(self) -> Dict[str, Any]:
        """Return context template.

        Returns:
            Example context
        """
        context_dict = schema_to_json(
            to_json_schema(infer(self.context_template_string))
        )
        context_dict["history"] = [
            schema_to_json(to_json_schema(infer(self.history_template_string)))
        ]
        return context_dict

    @abstractmethod
    def run(self, prompt: str, temperature: float = 1, topk: int = None) -> str:
        """Abstract method for concrete implementation of generation.

        Args:
            prompt: Fromatted prompt.
            temperature: Temperature parameter for sampling.
                Controls how random model output is: more temperature - more randomness
            topk: If not none selects top n of predictions to sample from during generation.

        Returns:
            Generated text
        """
        return None

    @abstractmethod
    def string_too_long(self, prompt: str) -> bool:
        """Check if prompt is too long.

        Args:
            prompt: Prompt to check.

        Returns:
            True if prompt is too long, False otherwise.
        """
        return False

    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """Return dictionary mapping for special tokens.

        To be implemented by child class.
        Can then be used in template string as fields
        Returns:
            Dictionary of special tokens
        """
        return None
