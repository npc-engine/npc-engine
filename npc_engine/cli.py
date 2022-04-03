#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the entry point for the command-line interface that starts npc-engine server."""
import sys
import os
import logging
import shutil

from npc_engine.exporters.base_exporter import Exporter

logging.basicConfig(level=logging.ERROR)

import click
from huggingface_hub import snapshot_download, hf_hub_download
from loguru import logger
import yaml

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
    model_manager.load_models()
    api_dict = model_manager.build_api_dict()
    rpc_server = ZMQServer(port)
    try:
        rpc_server.run(api_dict)
    except Exception:
        rpc_server.run(api_dict)


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"),
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
        snapshot_download(repo_id=model, revision="main", cache_dir=models_path)


@cli.command()
@click.option("--models-path", type=click.Path(exists=True))
def set_models_path(models_path: str):
    """Set the models path."""
    os.environ["NPC_ENGINE_MODELS_PATH"] = models_path


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"),
)
def list_models(models_path: str):
    """List the models in the folder."""
    from npc_engine.models.model_manager import ModelManager

    model_manager = ModelManager(models_path)
    model_manager.list_models()


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"),
)
@click.argument("model_id")
def download_model(models_path: str, model_id: str):
    """Download the model."""
    tmp_model_path = models_path + "/" + model_id.replace("/", "_")
    model_correct = True
    try:
        hf_hub_download(
            repo_id=model_id,
            filename="config.yml",
            cache_dir=tmp_model_path,
            force_filename="config.yml",
        )
        config_path = os.path.join(
            models_path, model_id.replace("/", "_"), "config.yml"
        )
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        if "model_type" not in config_dict:
            model_correct = False
    except ValueError:
        model_correct = False
    if model_correct:
        logger.info("Downloading model {}", model_id)
        os.remove(config_path)
        os.rmdir(tmp_model_path)
        snapshot_download(repo_id=model_id, revision="main", cache_dir=models_path)
    else:
        if click.confirm(
            click.style(
                f"{model_id} is not a valid npc-engine model."
                + " \nDo you want to export it?",
                fg="yellow",
            )
        ):
            export_model(models_path, model_id)


@cli.command()
@click.option(
    "--models-path",
    default=os.environ.get("NPC_ENGINE_MODELS_PATH", "./npc_engine/resources/models"),
)
@click.argument("model_id")
def export_model(models_path: str, model_id: str):
    """Export the model."""
    logger.info("Downloading source model {}", model_id)
    remove_source = False 
    if os.path.exists(model_id):
        source_path = model_id
    else:
        source_path = snapshot_download(
            repo_id=model_id, revision="main", cache_dir=models_path
        )
        remove_source = True
    export_path = (
        models_path + "/exported-" + model_id.replace("\\", "/").split("/")[-1]
    )
    os.mkdir(export_path)

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
def version():
    """Get the npc engine version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
