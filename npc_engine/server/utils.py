"""Utility functions for RPC communication."""
from typing import Any, Dict, Callable
import os
import subprocess

from platformdirs import user_cache_dir


def schema_to_json(
    s: Dict[str, Any], fill_value: Callable[[str], Any] = lambda _: ""
) -> Dict[str, Any]:
    """Iterate the schema and return simplified dictionary."""
    if "type" not in s and "anyOf" in s:
        return fill_value(s["title"])
    elif "type" in s and s["type"] == "object":
        return {k: schema_to_json(v) for k, v in s["properties"].items()}
    elif "type" in s and s["type"] == "array":
        return [schema_to_json(s["items"])]
    else:
        raise ValueError(f"Unknown schema type: {s}")


def start_test_server(port: str, models_path: str):  # pragma: no cover
    """Start the test server.

    Args:
        port: The port to start the server on.
        models_path: The path to the models.
    """
    subprocess.Popen(
        [
            "npc-engine",
            "--verbose",
            "run",
            "--port",
            port,
            "--models-path",
            models_path,
        ],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def build_ipc_uri(service_id: str) -> str:
    """Build ipc uri for the given service."""
    return f"ipc://{os.path.join(user_cache_dir('npc-engine'), service_id)}"
