"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from chatbot_server.text_generation import GPTTextGenerator
from chatbot_server.speech_synthesis import TacotronSpeechSynthesizer
import time
import logging


def test_total():
    """Check if chatbot works
    """
    chatbot_model = GPTTextGenerator(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\gpt')
    )
    chatbot_model.add_speaker(0, """
        I am old gray haired man
        I am a regular person
        I kill people for money
        I like chocolate cookies
    """)

    tts_module = TacotronSpeechSynthesizer(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models')
    )
    tts_module.create_voice(speaker_id=0)

    start = time.time()
    answer, trig = chatbot_model.step_dialog(0, "Hello my niggah")
    audio = tts_module.tts(0, answer)
    end = time.time()
    sa.play_buffer(audio, 1, 4, 22050)
    time.sleep(4)
    assert answer is not None
    print("done in {} seconds".format(end-start))
