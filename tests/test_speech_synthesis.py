"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from chatbot_server.speech_synthesis import TacotronSpeechSynthesizer
import time


def test_reply_default():
    """Check if voice with no traits works
    """
    tts_module = TacotronSpeechSynthesizer(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\tacotron')
    )
    test_line = "If you understand, test is successful"
    tts_module.create_voice(speaker_id="boii", voice_id=0)
    start = time.time()
    audio = tts_module.tts("boii", test_line)
    end = time.time()
    print(str(end - start) + ' seconds elapsed first run')
    start = time.time()
    audio = tts_module.tts("boii", test_line)
    end = time.time()
    print(str(end - start) + ' seconds elapsed second run')
    sa.play_buffer(audio, 1, 4, 22050)
    time.sleep(4)