API class is an abstract class that corresponds to a certain task a service should perform 
(e.g. text-to-speech or chatbot) and defines interface methods for such a task 
as well as abstract methods for specific servicess to implement.

All API classes are children of the BaseService class that handles registering service implementations and running them.

!!! note "Important"
    It also should list the methods that are to be exposed as API via API_METHODS class variable.

!!! note "Important"
    To be discovered correctly api classes must be imported into npc_engine.services module

## Existing APIs

These are the existing API classes and corresponding API_METHODS:

:::npc_engine.services.sequence_classifier.sequence_classifier_base.SequenceClassifierAPI
    selection:
        members:
            - compare
            - cache
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.chatbot.chatbot_base.ChatbotAPI
    selection:
        members:
            - generate_reply
            - get_context_fields
            - get_prompt_template
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.similarity.similarity_base.SimilarityAPI
    selection:
        members:
            - compare
            - cache
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.tts.tts_base.TextToSpeechAPI
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
from npc_engine.services.base_service import BaseService

class EchoAPI(BaseService):
    API_METHODS: List[str] = ["echo"]
    def __init__(self, *args, **kwargs):
        pass

    def echo(self, text):
        return text
```

!!! note "Dont forget"
    Import new API to npc-engine.services so that it is discovered. Models that are implemented for the API should appear there too. 

## Calling other APIs

There are a set of service clients that can be used to call other APIs that are running on the server. They will block other services so they should be used carefully.

:::npc_engine.service_clients
    rendering:
            heading_level: 3