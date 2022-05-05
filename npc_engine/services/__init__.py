"""Module that contains everything related to deep learning models.

For your model API to be discovered it must be imported here
"""

from .base_service import BaseService  # noqa: F401
from .tts import *  # noqa: F401,F403
from .text_generation import *  # noqa: F401,F403
from .similarity import *  # noqa: F401,F403
from .stt import *  # noqa: F401,F403
from .sequence_classifier import *  # noqa: F401,F403
