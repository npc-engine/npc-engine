import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    os.environ["NPC_ENGINE_MODELS_PATH"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources", "models"
    )
