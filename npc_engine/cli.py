#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the entry point for the command-line interface that starts npc-engine server."""
import os
import sys

import boto3
import click
from botocore.handlers import disable_signing
from loguru import logger

from npc_engine.models.model_manager import ModelManager
from npc_engine.version import __version__
from npc_engine.zmq_server import ZMQServer


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
    s3 = boto3.resource("s3")
    s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)  # Do not require credentials
    model_names = ["all-mini-lm-6-v2", "flowtron", "bart"]
    bucket = s3.Bucket("default-models")
    for model in model_names:
        logger.info('Downloading model {}', model)
        for file in bucket.objects.filter(Prefix=model):
            is_dir = file.key.split('/')[-1] == ''
            if is_dir:
                continue
            local_path = os.path.join(models_path, file.key)
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            logger.info('Downloading {}', file.key)
            bucket.download_file(file.key, local_path)


@cli.command()
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
