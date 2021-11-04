#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the entry point for the command-line interface that starts npc-engine server.

.. currentmodule:: inference_engine.cli
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>
"""
import logging
import click
import os
import zmq
from inference_engine.version import __version__
from inference_engine.inference_engine import InferenceEngine
import time


LOGGING_LEVELS = {
    0: logging.NOTSET,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}  #: a mapping of `verbose` option counts to logging levels


class Info(object):
    """An information object to pass data between CLI functions."""

    def __init__(self):  # Note: This object must have an empty constructor.
        """Create a new instance."""
        self.verbose: int = 0


# pass_info is a decorator for functions that pass 'Info' objects.
#: pylint: disable=invalid-name
pass_info = click.make_pass_decorator(Info, ensure=True)


@click.group()
@click.option("--verbose", "-v", count=True, help="Enable verbose output.")
@pass_info
def cli(info: Info, verbose: int):
    """Run inference_engine."""
    # Use the verbosity count to determine the logging level...
    if verbose > 0:
        logging.basicConfig(
            level=LOGGING_LEVELS[verbose]
            if verbose in LOGGING_LEVELS
            else logging.DEBUG
        )
        logging.getLogger().setLevel(
            LOGGING_LEVELS[verbose] if verbose in LOGGING_LEVELS else logging.DEBUG
        )
        click.echo(
            click.style(
                f"Verbose logging is enabled. "
                f"(LEVEL={logging.getLogger().getEffectiveLevel()})",
                fg="yellow",
            )
        )
    info.verbose = verbose


@cli.command()
@click.option("--models-path", default="./Assets/StreamingAssets")
def run(models_path: str):
    print("starting server")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    bart_model = os.path.join(models_path, "bart")
    tacotron = os.path.join(models_path, "flowtron_squeezewave")
    roberta_semb = os.path.join(models_path, "roberta_semb")

    chatbot = InferenceEngine(bart_model, tacotron, roberta_semb)

    try:
        while True:
            #  Wait for next request from client
            message = socket.recv_json()
            logging.info("Received request: %s" % message)

            start = time.time()
            reply = chatbot.handle_message(message)
            end = time.time()

            logging.info("Handle message time: %d" % (end - start))
            logging.info("Message reply: %s" % (reply))
            #  Send reply back to client
            socket.send_json(reply)
    except Exception as e:
        logging.error(e)


@cli.command()
def version():
    """Get the library version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
