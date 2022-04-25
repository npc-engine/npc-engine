Model classes are specific implementations of API classes. 
They define the loading process in `__init__` function and 
all the abstract methods required by the API class to function.

## How models are loaded?

[ModelManager](../reference/#npc_engine.models.model_manager.ModelManager) Scans the models folder. For each discovered subfolder [Model.load](../reference/#npc_engine.models.base_model.Model.load) is called, which tries to read `config.yml` field `model_type`. This field must contain correct model class that was discovered and registered by `Model` parent class, and if it does, this class is instantiated with whole `config.yml` parsed dictionary as parameters. 

## How is their API exposed?

[ModelManager](../reference/#npc_engine.models.model_manager.ModelManager) 
builds a mapping from model's API_METHODS class variable to anonymous functions that call these methods on the model. This dictionary is then served through `json-rpc` protocol implementation. 

## Existing model classes

:::npc_engine.models.chatbot.bart.BartChatbot
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.models.similarity.similarity_transformers.TransformerSemanticSimilarity
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.models.tts.flowtron.FlowtronTTS
    rendering:
      show_root_heading: true
      show_source: false


## Creating new models

You can use this dummy model example to create your own:

```python
from npc_engine.models.base_model import Model

class EchoModel(ChatbotAPI):

    def __init__(self, model_path:str, *args, **kwargs):
        print("model is in {model_path}")
    
    def get_special_tokens(self):
        return {}

    def run(self, prompt, temperature=1, topk=None):
        return prompt
```

!!! note "Dont forget"
    Import new model to npc-engine.models so that it is discovered. 