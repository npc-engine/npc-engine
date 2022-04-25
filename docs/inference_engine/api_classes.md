API class is an abstract class that corresponds to a certain task a model should perform 
(e.g. text-to-speech or chatbot) and defines interface methods for such a task 
as well as abstract methods for specific models to implement.

All API classes are children of the Model class that handles registering model implementations and loading them.

!!! note "Important"
    It also should list the methods that are to be exposed as API via API_METHODS class variable.

!!! note "Important"
    To be discovered correctly api classes must be imported into npc_engine.models module

## Existing APIs

These are the existing API classes and corresponding API_METHODS:

:::npc_engine.models.chatbot.chatbot_base.ChatbotAPI
    selection:
        members:
            - generate_reply
            - get_context_fields
            - get_prompt_template
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.models.similarity.similarity_base.SimilarityAPI
    selection:
        members:
            - compare
            - cache
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.models.tts.tts_base.TextToSpeechAPI
    selection:
        members:
            - tts_start
            - tts_get_results
            - get_speaker_ids
    rendering:
      show_root_heading: true
      show_source: false

## Creating new APIs

You can use this dummy API example to create your own:

```python
from npc_engine.models.base_model import Model

class EchoAPI(Model):
    API_METHODS: List[str] = ["echo"]
    def __init__(self, *args, **kwargs):
        pass

    def echo(self, text):
        return text
```

!!! note "Dont forget"
    Import new API to npc-engine.models so that it is discovered. Models that are implemented for the API should appear there too. 
