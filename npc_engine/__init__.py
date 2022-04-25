#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Server for providing onnx runtime predictions for text generation and speech synthesis.

Uses 0MQ REP/REQ sockets with JSONRPC 2.0 protocol.
"""

from .version import __version__, __release__  # noqa
