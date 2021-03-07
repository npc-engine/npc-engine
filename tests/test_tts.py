"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from inference_engine.speech_synthesis import TTS
import time


def test_reply_default():
    """Check if voice with no traits works
    """
    tts_module = TTS(
        os.path.join(
            os.path.dirname(__file__),
            "..\\inference_engine\\resources\\models\\flowtron_squeezewave",
        )
    )
    test_line = "Hello friend my name is my name is my name is slim shady"
    start = time.time()
    audio = tts_module.tts(6, test_line)
    end = time.time()
    print(str(end - start) + " seconds elapsed")
    sa.play_buffer(audio, 1, 4, 22050)
    time.sleep(4)
