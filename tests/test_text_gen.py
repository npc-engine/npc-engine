"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from chatbot_server.text_generation import GPTTextGenerator
import time
import logging


def test_reply_default():
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
    start = time.time()
    answer = chatbot_model.step_dialog(0, "Hello my niggah")
    end = time.time()
    assert answer is not None
    print("done in {} seconds".format(end-start))
