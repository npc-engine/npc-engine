"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from chatbot_server.chatbot import Chatbot
import time
import logging


def test_run():
    """Check if chatbot works
    """
    models_path = os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models')
    gpt_model = os.path.join(models_path, "gpt")
    tacotron = os.path.join(models_path, "tacotron")
    roberta_semb = os.path.join(models_path, "roberta_semb")

    chatbot = Chatbot(gpt_model, tacotron, roberta_semb)

    create_speaker_msg = {
        'cmd': "create_speaker",
        'speaker_id': "test_speaker",
        'persona': "Test persona.", 
        'temperature': 0.85, 
        'traits': ['1']
    }

    script_line_msg = {        
        'cmd': "script_line",
        'speaker_id': "test_speaker",
        'cue_lines': ["Test cue"],
        'script_lines': ["Test response"],
        'parent': "root",
        'node_id': "test_node",
        'expires_after': 5,
        'threshold': 0.6
    }

    step_dialog_msg = {
        'cmd': "step_dialog",
        'speaker_id': "test_speaker", 
        'line': "Hello world"
    }
    start = time.time()

    resp = chatbot.handle_message(create_speaker_msg)
    assert resp['status'] == 0 
    
    resp = chatbot.handle_message(script_line_msg)
    assert resp['status'] == 0 
    
    resp = chatbot.handle_message(step_dialog_msg)
    assert resp['status'] == 0
    assert resp['reply'] is not None
    assert resp['reply_text'] is not None
    assert resp['script_triggered'] is None
    
    end = time.time()

    print("done in {} seconds".format(end-start))
