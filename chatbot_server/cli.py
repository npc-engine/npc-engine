#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the entry point for the command-line interface (CLI) application.

It can be used as a handy facility for running the task from a command line.

.. note::

    To learn more about Click visit the
    `project website <http://click.pocoo.org/5/>`_.  There is also a very
    helpful `tutorial video <https://www.youtube.com/watch?v=kNke39OZ2k0>`_.

    To learn more about running Luigi, visit the Luigi project's
    `Read-The-Docs <http://luigi.readthedocs.io/en/stable/>`_ page.

.. currentmodule:: chatbot_server.cli
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>
"""
import logging
import click
import zmq
from chatbot_server.version import __version__
from chatbot_server.chatbot import Chatbot
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


# Change the options to below to suit the actual options for your task (or
# tasks).
@click.group()
@click.option("--verbose", "-v", count=True, help="Enable verbose output.")
@pass_info
def cli(info: Info, verbose: int):
    """Run chatbot_server."""
    # Use the verbosity count to determine the logging level...
    if verbose > 0:
        logging.basicConfig(
            level=LOGGING_LEVELS[verbose]
            if verbose in LOGGING_LEVELS
            else logging.DEBUG
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
@click.option("--gpt-model", default="./gpt2.onnx")
@click.option("--tacotron2", default="./gpt2.onnx")
def run(gpt_model: str, tacotron2: str):
    print("starting server")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    # session = rt.InferenceSession(gpt_model, providers=["CUDAExecutionProvider"]) # Noqa:
    chatbot = Chatbot(gpt_model, tacotron2)

    try:
        while True:
            #  Wait for next request from client
            message = socket.recv_json()
            logging.warning("Received request: %s" % message)

            start = time.time()
            reply = chatbot.handle_message(message)
            end = time.time()

            logging.warning("Handle message time: %d" % (end-start))
            #  Send reply back to client
            socket.send_json(reply)
    except Exception as e:
        logging.error(e)
        exit(1)


@cli.command()
def version():
    """Get the library version."""
    click.echo(click.style(f"{__version__}", bold=True))


if __name__ == "__main__":
    cli()
