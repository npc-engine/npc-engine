#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the entry point for the command-line interface that starts npc-engine server."""
import sys
import os
import logging

logging.basicConfig(level=logging.ERROR)

import click
from huggingface_hub import snapshot_download
from loguru import logger

from npc_engine.version import __version__


@click.group()
@click.option("--verbose", "-v", default=False, help="Enable verbose output.")
def cli(verbose: bool):
    """NPC engine JSON RPC server CLI."""
    # Use the verbosity count to determine the logging level...
    if verbose:
        logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
        click.echo(
            click.style("Verbose logging is enabled. (LEVEL=INFO)", fg="yellow",)
        )

@cli.command()
@click.option("--models-path", default="./npc_engine/resources/models")
@click.option("--port", default="5556")
def run(models_path: str, port: str):
    """Load the models and start JSONRPC server."""
    from npc_engine.models.model_manager import ModelManager
    from npc_engine.zmq_server import ZMQServer
    model_manager = ModelManager(models_path)
    api_dict = model_manager.build_api_dict()
    rpc_server = ZMQServer(port)
    try:
        rpc_server.run(api_dict)
    except Exception:
        rpc_server.run(api_dict)


@cli.command()
@click.option("--models-path", default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"))
def download_default_models(models_path: str):
    """Download default models into the folder."""
    model_names = [
        "npc-engine/exported-paraphrase-MiniLM-L6-v2",
        "npc-engine/exported-bart-light-gail-chatbot",
        "npc-engine/exported-nemo-quartznet-ctc-stt",
        "npc-engine/exported-flowtron-waveglow-librispeech-tts",
    ]
    for model in model_names:
        logger.info("Downloading model {}", model)
        logger.info("Downloading {}", model)
        snapshot_download(repo_id=model, revision="main", cache_dir=models_path)


@cli.command()
@click.option("--models-path", type=click.Path(exists=True))
def set_models_path(models_path: str):
    """Set the models path."""
    os.environ["NPC_ENGINE_MODELS_PATH"] = models_path


@cli.command()
@click.option("--models-path", default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"))
def list_models(models_path: str):
    """List the models in the folder."""
    from npc_engine.models.model_manager import ModelManager
    model_manager = ModelManager(models_path)
    model_manager.list_models()


@cli.command()
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
