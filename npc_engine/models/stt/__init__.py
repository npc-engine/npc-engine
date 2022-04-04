# -*- coding: utf-8 -*-
"""Speech to text API.

This module implements specific models and wraps them under
the common interface for loading and inference.

Example:
    ```
        from npc_engine.models.stt import SpeechToTextAPI
        model = SpeechToTextAPI.load("path/to/model_dir")
        text = model.listen()  # Say something
    ```
"""

from .stt_base import SpeechToTextAPI  # noqa: F401
from .nemo_stt import NemoSTT  # noqa: F401


# pyinstaller hidden modules
import sklearn.utils._cython_blas  # noqa: F401
import sklearn.utils._typedefs  # noqa: F401
import sklearn.neighbors._partition_nodes  # noqa: F401
import sklearn.tree._utils  # noqa: F401
