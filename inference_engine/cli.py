#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the entry point for the command-line interface that starts npc-engine server.

.. currentmodule:: inference_engine.cli
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>
"""
import click
import sys
from inference_engine.models.model_manager import ModelManager
from inference_engine.version import __version__
from inference_engine.zmq_server import ZMQServer
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
@click.option("--models-path", default="./Assets/StreamingAssets")
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
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
