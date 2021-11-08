from typing import Dict, Any, List

from abc import abstractmethod
from inference_engine.models.base_model import Model
from jinja2 import Template
import json


class Chatbot(Model):
    """Abstract base class for Chatbot models.
    """

    def __init__(self, template_string, default_context, *args, **kwargs):
        self.template_string = template_string
        self.default_context = json.loads(default_context)
        self.template = Template(template_string)
        self.initialized = True

    def generate_reply(self, context: Dict[str, Any], *args, **kwargs) -> str:
        """Formats model prompt and generates response.

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
        prompt = self.template.render(**context)
        return self.run(prompt, *args, **kwargs)

    @abstractmethod
    def run(self, prompt: str, temperature: float, topk: int = None) -> str:
        """Abstract method for concrete implementation of generation.

        Args:
            prompt: Fromatted prompt.
            temperature: Temperature parameter for sampling.
                Controls how random model output is: more temperature - more randomness
            topk: If not none selects top n of predictions to sample from during generation.

        Returns:
            Generator that yields next chunk of speech in the form of f32 ndarray.
        """
        return None

    def get_context_fields(self) -> List[str]:
        """Returns context template used for formatting model prompt

        Returns:
            A template context dict with empty fields.
        """
        return self.default_context

    def get_prompt_template(self) -> str:
        """Returns prompt template string used to render model prompt.

        Returns:
            A template string.
        """
        return self.template_string
