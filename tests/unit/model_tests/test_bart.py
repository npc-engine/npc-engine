"""Text generation test."""
import os
from npc_engine.services import BaseService
import inspect
import sys

from npc_engine.services.utils.config import get_type_from_dict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import mocks.zmq_mocks as zmq
import yaml

path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "models")

subdirs = [
    f.path
    for f in os.scandir(path)
    if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
]

configs = [
    yaml.safe_load(open(os.path.join(subdir, "config.yml"), "r")) for subdir in subdirs
]

bart_paths = [
    subdir
    for config, subdir in zip(configs, subdirs)
    if "BartChatbot" in get_type_from_dict(config)
]


def test_reply_default():
    """Check if chatbot works"""
    chatbot_model = BaseService.create(
        zmq.Context(), bart_paths[0], service_id="test", uri="test"
    )

    print(f"Special tokens {chatbot_model.get_special_tokens()}")

    answer = chatbot_model.generate_reply(
        context=dict(
            location_name="Brimswood pub, Tavern",
            location="""The Brimswood pub is an old establishment. 
It is sturdy, has a lot of life in its walls, but hasn't been updated in decades.
The clientele are the same as they always are, and they don't see very many strangers. 
The vibe is somber, and conversations are usually had in hushed tones.""",
            name="pet dog",
            persona="""I am mans best friend and I wouldn't have it any other way. I tend to my master and never leave his side. 
I sleep at his feet and guard the room at night from things that go bump in the night.""",
            other_name="the town baker's husband",
            other_persona="""I am the town baker's husband and I love eating pastries.  
I tend to be in very good spirits and enjoy selling delicious baked goods that my wife has made.  
My wife is great at baking but she is lousy at washing my clothes.  
They keep shrinking!""",
            history=[
                {"speaker": "the town baker's husband", "line": "Hello friend!"},
                {"speaker": "pet dog", "line": "Hello!"},
            ],
        ),
        temperature=0.8,
        topk=None,
    )
    assert answer is not None
