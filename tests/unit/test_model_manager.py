"""Model manager test."""
import os
from npc_engine import models

from npc_engine.models.model_manager import ModelManager
import pytest


def test_model_manager_api_dict():
    """Test if all api methods are registered"""
    model_manager = ModelManager(
        os.path.join(os.path.dirname(__file__), "..", "resources", "models")
    )
    model_manager.load_models()
    api_dict = model_manager.build_api_dict()
    assert len(api_dict) > 0


def test_model_manager_list_models():
    """Test if models are printed without error."""
    model_manager = ModelManager(
        os.path.join(os.path.dirname(__file__), "..", "resources", "models")
    )

    model_manager.list_models()
