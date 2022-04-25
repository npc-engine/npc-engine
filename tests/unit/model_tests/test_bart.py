"""Text generation test."""
import os
from npc_engine.models import Model
import time
import pytest


model_fp = os.path.join(
    os.path.dirname(__file__), "..\\..\\..\\npc_engine\\resources\\models\\bart",
)


@pytest.mark.skipif(
    not os.path.exists(model_fp), reason="Model missing",
)
def test_reply_default():
    """Check if chatbot works"""
    chatbot_model = Model.load(model_fp)

    start = time.time()
    answer = chatbot_model.generate_reply(
        context=dict(
            persona="""
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
            history=["<speaker_other>Hello friend!"],
        ),
        temperature=0.8,
        topk=None,
    )
    end = time.time()
    assert answer is not None
    print("Answer: {}".format(answer))
    print("done in {} seconds".format(end - start))
