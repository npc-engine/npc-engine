"""Model manager test."""
import os
from npc_engine import models

from npc_engine.models.model_manager import ModelManager
import pytest


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "npc_engine", "resources", "models"
        )
    ),
    reason="Models Folder missing",
)
def test_model_manager_api_dict():
    """Test if all api methods are registered"""
    model_manager = ModelManager(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "npc_engine", "resources", "models"
        )
    )

    api_dict = model_manager.build_api_dict()
