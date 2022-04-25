"""Model manager test."""
import os
from npc_engine import models

from npc_engine.models.model_manager import ModelManager


def test_model_manager_api_dict():
    """Test if all api methods are registered"""
    model_manager = ModelManager(
        os.path.join(os.path.dirname(__file__), "..\\..\\npc_engine\\resources\\models")
    )

    api_dict = model_manager.build_api_dict()
    methods = [
        "generate_reply",
        "get_context_fields",
        "get_prompt_template",
        "tts_start",
        "tts_get_results",
        "get_speaker_ids",
        "compare",
        "cache",
    ]
    print(api_dict)
    api_dict["compare"](*["I will help you", ["I shall provide you my assistance"]])
    for method in methods:
        assert method in api_dict
