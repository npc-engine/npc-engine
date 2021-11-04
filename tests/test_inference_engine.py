"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from inference_engine.inference_engine import InferenceEngine
import time
import logging


def test_run():
    """Check if chatbot works
    """
    models_path = os.path.join(
        os.path.dirname(__file__), "..\\inference_engine\\resources\\models"
    )
    bart_model = os.path.join(models_path, "bart")
    tacotron = os.path.join(models_path, "flowtron_squeezewave")
    roberta_semb = os.path.join(models_path, "roberta_semb")

    inference_engine = InferenceEngine(bart_model, tacotron, roberta_semb)

    tts_msg = {"cmd": "start_tts", "voice_id": 0, "line": "Hello this is a test"}
    next_tts_msg = {"cmd": "tts_next"}

    semantic_test_add = {
        "cmd": "add_test",
        "test_id": "name test",
        "lines": ["My name is", "I am ", "Name is"],
    }

    semantic_test = {
        "cmd": "test",
        "test_ids": ["name test"],
        "line": "My name is Jeff",
        "method": "OR",
    }

    semantic_test_custom = {
        "cmd": "test",
        "query_lines": ["My name is", "I am ", "Name is"],
        "line": "My name is Jeff",
        "method": "OR",
    }

    chatbot_message = {
        "cmd": "chatbot",
        "persona": """
            _setting_name Brimswood pub, Tavern
            _setting_desc The Brimswood pub is an old establishment. 
            It is sturdy, has a lot of life in its walls, but hasn't been updated in decades.
            The clientele are the same as they always are, and they don't see very many strangers. 
            The vibe is somber, and conversations are usually had in hushed tones.</s>
            <speaker_self>_self_name pet dog
            _self_persona I am mans best friend and I wouldn't have it any other way. I tend to my master and never leave his side. 
            I sleep at his feet and guard the room at night from things that go bump in the night.</s>
            <speaker_other>_partner_name the town baker's husband
            _other_persona I am the town baker's husband and I love eating pastries.  
            I tend to be in very good spirits and enjoy selling delicious baked goods that my wife has made.  
            My wife is great at baking but she is lousy at washing my clothes.  
            They keep shrinking!
        """.strip(),
        "history": [],
        "temperature": 0.8,
        "topk": 75,
    }
    start = time.time()

    resp = inference_engine.handle_message(next_tts_msg)
    assert resp["status"] != "OK"

    resp = inference_engine.handle_message(tts_msg)
    assert resp["status"] == "OK"

    resp = inference_engine.handle_message(next_tts_msg)
    assert len(resp["audio"]) > 0

    resp = inference_engine.handle_message(semantic_test_add)
    assert resp["status"] == "OK"

    resp = inference_engine.handle_message(semantic_test)
    assert resp["status"] == "OK"

    resp_custom = inference_engine.handle_message(semantic_test_custom)
    assert resp["status"] == "OK"
    assert abs(resp["results"]["name test"] - resp_custom["results"]) <= 1e-4

    resp = inference_engine.handle_message(chatbot_message)
    assert resp["status"] == "OK"
    assert resp["reply"] is not None
    end = time.time()
    print("reply", resp["reply"])

    print("done in {} seconds".format(end - start))
