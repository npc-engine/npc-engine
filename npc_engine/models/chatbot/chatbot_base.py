"""Module that implements chatbot model API."""
from typing import Dict, Any, List

from abc import abstractmethod
from npc_engine.models.base_model import Model
from jinja2 import Template
import json


class ChatbotAPI(Model):
    """Abstract base class for Chatbot models."""

    API_METHODS: List[str] = [
        "generate_reply",
        "get_context_fields",
        "get_prompt_template",
    ]

    def __init__(self, template_string: str, default_context: str, *args, **kwargs):
        """Initialize prompt formatting variables.

        Args:
            template_string: Template string to be rendered.
            default_context: Context example with empty fields.
        """
        self.template_string = template_string
        self.default_context = json.loads(default_context)
        self.template = Template(template_string)
        self.initialized = True

    def generate_reply(self, context: Dict[str, Any], *args, **kwargs) -> str:
        """Format the model prompt and generates response.

        Args:
            context: Prompt context.
            *args
            **kwargs

        Returns:
            Text response to a prompt.
        """
        if not self.initialized:
            raise AssertionError(
                "Can not generate replies before base Chatbot class was initialized"
            )
        prompt = self.template.render(**context, **self.get_special_tokens())
        return self.run(prompt, *args, **kwargs)

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
    def get_special_tokens(self) -> Dict[str, str]:
        """Return dictionary mapping for special tokens.

        To be implemented by child class.
        Can then be used in template string as fields
        Returns:
            Dictionary of special tokens
        """
        return None

    def get_context_fields(self) -> List[str]:
        """Return context template used for formatting model prompt.

        Returns:
            A template context dict with empty fields.
        """
        return self.default_context

    def get_prompt_template(self) -> str:
        """Return prompt template string used to render model prompt.

        Returns:
            A template string.
        """
        return self.template_string
