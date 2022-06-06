#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the entry point for the command-line interface that starts npc-engine server."""
import sys
import os
import logging


logging.basicConfig(level=logging.ERROR)
import shutil
import zmq
import zmq.asyncio
from npc_engine.services.utils.config import (
    get_model_type_name,
    validate_hub_model,
    validate_local_model,
)
from npc_engine.server.metadata_manager import MetadataManager


import click
from huggingface_hub import snapshot_download
from loguru import logger

from npc_engine.version import __version__
from multiprocessing import freeze_support


@click.group()
@click.option("--verbose/--silent", "-v", default=False, help="Enable verbose output.")
def cli(verbose: bool):
    """NPC engine JSON RPC server CLI."""
    # Use the verbosity count to determine the logging level...
    logger.remove()
    if verbose:
        logger.add(
            sys.stdout, format="{time} {level} {message}", level="DEBUG", enqueue=True
        )
        logger.add(
            os.path.join("Logs", "npc-engine.log"),
            format="{time} {level} {message}",
            level="DEBUG",
            enqueue=True,
            rotation="500 MB",
        )
        click.echo(
            click.style("Verbose logging is enabled. (LEVEL=INFO)", fg="yellow",)
        )
    else:
        logger.add(
            sys.stdout, format="{time} {level} {message}", level="WARNING", enqueue=True
        )
        logger.add(
            os.path.join("Logs", "npc-engine.log"),
            format="{time} {level} {message}",
            level="WARNING",
            enqueue=True,
            rotation="500 MB",
        )


@cli.command()
@click.option("--port", default="5555", help="The port to listen on.")
@click.option(
    "--start-all/--dont-start",
    default=True,
    help="Whether to start all services or not.",
)
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs",
)
@click.option("--http/--zmq", default=False, help="Whether to use HTTP or ZMQ.")
def run(port: str, start_all: bool, models_path: str, http: bool):
    """Load the models and start JSONRPC server."""
    from npc_engine.server.control_service import ControlService

    context = zmq.asyncio.Context(io_threads=5)
    metadata_manager = MetadataManager(models_path, port)
    metadata_manager.port = port
    control_service = ControlService(context, metadata_manager)

    if not http:
        from npc_engine.server.server import ZMQServer

        server = ZMQServer(context, control_service, metadata_manager, start_all)
    else:
        from npc_engine.server.server import HTTPServer

        server = HTTPServer(context, control_service, metadata_manager, start_all)
    server.run()


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs.",
)
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
        tmp_folder = snapshot_download(
            repo_id=model, revision="main", cache_dir=models_path
        )
        os.rename(tmp_folder, os.path.join(models_path, model.split("/")[-1]))


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs.",
)
def set_models_path(models_path: str):
    """Set the default models path.

    Args:
        models_path (str): The path to the models.
    """
    os.environ["NPC_ENGINE_MODELS_PATH"] = models_path


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs.",
)
def list_models(models_path: str):
    """List the models in the folder."""
    metadata_manager = MetadataManager(models_path, "not_used")
    metadata_list = metadata_manager.get_services_metadata()
    for metadata in metadata_list:
        click.echo(metadata["id"])
        click.echo(metadata["service"])
        click.echo("Service description:")
        click.echo(metadata["service_short_description"])
        click.echo("Model description:")
        click.echo(metadata["readme"].split("\n\n")[0])
        click.echo("--------------------")


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs.",
)
@click.argument("model_id")
def describe(models_path: str, model_id: str):
    """Show service detailed information.

    model_id argument follows service resolution rules of the npc-engine.
    """
    model_manager = MetadataManager(models_path, "not_used")
    metadata = model_manager.get_metadata(model_id)

    click.echo(metadata["id"])
    click.echo(metadata["type"])
    click.echo("Service description:")
    click.echo(metadata["service_description"])
    click.echo("Model description:")
    click.echo(metadata["readme"])


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models",),
    help="The path to the folder with service configs.",
)
@click.argument("model_id")
def download_model(models_path: str, model_id: str):
    """Download a model from Huggingface Hub."""
    model_correct = validate_hub_model(models_path, model_id)
    if model_correct:
        logger.info("Downloading model {}", model_id)
        tmp_folder = snapshot_download(
            repo_id=model_id, revision="main", cache_dir=models_path
        )
        os.rename(tmp_folder, os.path.join(models_path, model_id.split("/")[-1]))
    else:
        if click.confirm(
            click.style(
                f"{model_id} is not a valid npc-engine model."
                + " \nDo you want to export it?",
                fg="yellow",
            )
        ):
            export_model(models_path, model_id, True)


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models",),
    help="The path to the folder with service configs.",
)
@click.argument("model_id")
def export_model(models_path: str, model_id: str, remove_source: bool = False):
    """Export the model."""
    from npc_engine.exporters.base_exporter import Exporter

    logger.info("Downloading source model {}", model_id)
    if os.path.exists(model_id):
        source_path = model_id
    else:
        source_path = snapshot_download(
            repo_id=model_id, revision="main", cache_dir=models_path
        )
        remove_source = True
    export_path = os.path.join(
        models_path, "exported-" + model_id.replace("\\", "/").split("/")[-1],
    )
    os.makedirs(export_path, exist_ok=True)

    logger.info("Exporting model {} to {}", model_id, export_path)
    exporters = Exporter.get_exporters()
    click.echo("Available exporters:")
    for i, exporter in enumerate(exporters):
        click.echo(f"{i+1}. {exporter.description()}")
    exporter_id = click.prompt("Please select an exporter", type=int)
    exporter = exporters[exporter_id - 1]
    exporter.export(source_path, export_path)
    exporter.create_config(export_path)
    if remove_source:
        shutil.rmtree(source_path)


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./models"),
    help="The path to the folder with service configs.",
)
@click.argument("model_id")
def test_model(models_path: str, model_id: str):
    """Send test request to the model and print reply."""
    from npc_engine.exporters.base_exporter import Exporter

    if not validate_local_model(models_path, model_id):
        click.echo(
            click.style(f"{(model_id)} is not a valid npc-engine model.", fg="red",)
        )
        return 1
    model_type = get_model_type_name(models_path, model_id)
    exporters = Exporter.get_exporters()
    for exporter in exporters:
        if exporter.get_model_name() == model_type:
            exporter.test_model(models_path, model_id)
            return 0


@cli.command()
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    freeze_support()
    cli()
