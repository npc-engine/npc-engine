"""Config related functions."""
import os
import yaml
from huggingface_hub import hf_hub_download
from requests.exceptions import HTTPError


def get_type_from_dict(config_dict: dict) -> str:
    """Get model type from config dict.

    Args:
        config_dict: Config dict.

    Returns:
        Model type.
    """
    return config_dict.get("model_type", config_dict.get("type", ""))


def get_model_type_name(models_path: str, model_id: str) -> str:
    """Get model type name.

    Args:
        models_path: Path to the models folder.
        model_id: Model id (dirname).

    Returns:
        Model type name.
    """
    model_path = os.path.join(models_path, model_id)
    config_path = os.path.join(model_path, "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return get_type_from_dict(config)


def validate_model(models_path: str, model_id: str) -> bool:
    """Validate model by id.

    Args:
        models_path: Path to the models folder.
        model_id: Model id.
    """
    model_correct = True
    model_path = os.path.join(models_path, model_id)
    if not os.path.exists(model_path):
        model_correct = validate_hub_model(models_path, model_id)
    else:
        model_correct = validate_local_model(models_path, model_id)
    return model_correct


def validate_local_model(models_path: str, model_id: str) -> bool:
    """Validate local model by id.

    Args:
        models_path: Path to the models folder.
        model_id: Model id.
    """
    model_correct = True
    model_path = os.path.join(models_path, model_id)
    if not os.path.exists(model_path):
        model_correct = False
    else:
        try:
            _ = get_model_type_name(models_path, model_id)
        except FileNotFoundError:
            model_correct = False
    return model_correct


def validate_hub_model(models_path: str, model_id: str) -> bool:
    """Validate huggingface hub model by id.

    Args:
        models_path: Path to the models folder.
        model_id: Huggingface hub model id.
    """
    tmp_model_path = os.path.join(models_path, model_id.replace("/", "_"))
    model_correct = True
    try:
        try:
            hf_hub_download(
                repo_id=model_id,
                filename="config.yml",
                cache_dir=tmp_model_path,
                force_filename="config.yml",
            )
        except HTTPError:
            return False
        config_path = os.path.join(
            models_path, model_id.replace("/", "_"), "config.yml"
        )
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        if "model_type" not in config_dict:
            model_correct = False
    except ValueError:
        model_correct = False

    if os.path.exists(config_path):
        os.remove(config_path)
    if os.path.exists(tmp_model_path):
        os.rmdir(tmp_model_path)
    return model_correct
