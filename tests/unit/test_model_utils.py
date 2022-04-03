"""models module utils tests."""
import os


def test_validate_hub_model():
    """Test if validate_hub_model works."""
    from npc_engine.models.utils import validate_hub_model

    models_path = os.path.join(os.path.dirname(__file__), ".")
    model_id = "npc-engine/exported-paraphrase-MiniLM-L6-v2"
    assert validate_hub_model(models_path, model_id)


def test_validate_hub_model_invalid():
    """Test if validate_hub_model works."""
    from npc_engine.models.utils import validate_hub_model

    models_path = os.path.join(os.path.dirname(__file__), ".")
    model_id = "bert-base-cased"
    assert not validate_hub_model(models_path, model_id)


def test_validate_model():
    """Test if validate_model works."""
    from npc_engine.models.utils import validate_model

    models_path = os.path.join(os.path.dirname(__file__), ".")
    model_id = "npc-engine/exported-paraphrase-MiniLM-L6-v2"
    assert validate_model(models_path, model_id)


def test_validate_model_invalid():
    """Test if validate_model works."""
    from npc_engine.models.utils import validate_model

    models_path = os.path.join(os.path.dirname(__file__), ".")
    model_id = "bert-base-cased"
    assert not validate_model(models_path, model_id)
