#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the entry point for the command-line interface that starts npc-engine server."""
import click
import sys
import os
import pathlib
from npc_engine.models.model_manager import ModelManager
from npc_engine.version import __version__
from npc_engine.zmq_server import ZMQServer
from loguru import logger


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
    model_manager = ModelManager(models_path)
    api_dict = model_manager.build_api_dict()
    rpc_server = ZMQServer(port)
    try:
        rpc_server.run(api_dict)
    except Exception:
        rpc_server.run(api_dict)


@cli.command()
@click.option("--models-path", default="./npc_engine/resources/models")
def download_default_models(models_path: str):
    """Download default models into the folder."""
    pathlib.Path(os.path.join(models_path, "flowtron")).mkdir(
        parents=True, exist_ok=True
    )
    os.system(
        f"cd {models_path} "
        + "&& gdown --folder https://drive.google.com/drive/folders/1JHu23RrnUHO8eLXiAP08fkg6H_1MhgBM?usp=sharing"
    )
    pathlib.Path(os.path.join(models_path, "all-mini-lm-6-v2")).mkdir(
        parents=True, exist_ok=True
    )
    os.system(
        f"cd {models_path} "
        + "&& gdown --folder https://drive.google.com/drive/folders/1KZKNe3PdEnqbRoS3U64U4XW7ndR-sTAm?usp=sharing"
    )
    pathlib.Path(os.path.join(models_path, "bart")).mkdir(parents=True, exist_ok=True)
    os.system(
        f"cd {models_path} "
        + "&& gdown --folder https://drive.google.com/drive/folders/1Vk2rqVeOaE48dP3HhHa5g4N-CHxh0fQn?usp=sharing"
    )


@cli.command()
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
