Model classes are specific implementations of API classes. 
They define the loading process in `__init__` function and 
all the abstract methods required by the API class to function.

## How models are loaded?

[ModelManager](../reference/#npc_engine.models.model_manager.ModelManager) Scans the models folder. For each discovered subfolder [Model.load](../reference/#npc_engine.models.base_model.Model.load) is called, which tries to read `config.yml` field `model_type`. This field must contain correct model class that was discovered and registered by `Model` parent class, and if it does, this class is instantiated with whole `config.yml` parsed dictionary as parameters. 

## How is their API exposed?

[ModelManager](../reference/#npc_engine.models.model_manager.ModelManager) 
builds a mapping from model's API_METHODS class variable to anonymous functions that call these methods on the model. This dictionary is then served through `json-rpc` protocol implementation. 

## Existing service classes

:::npc_engine.services.sequence_classifier.hf_classifier.HfClassifier
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.chatbot.hf_chatbot.HfChatbot
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.chatbot.bart.BartChatbot
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.similarity.similarity_transformers.TransformerSemanticSimilarity
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.tts.flowtron.FlowtronTTS
    rendering:
      show_root_heading: true
      show_source: false

## Default Models

- ### Fantasy Chatbot

    [BartChatbot](../reference/#npc_engine.services.chatbot.chatbot_base.ChatbotAPI)
    trained on [LIGHT Dataset](https://parl.ai/projects/light/). 
    Model consumes both self, other personas and location dialogue is happening in.

<!-- TODO: Change context to better reflect the required arguments, describe context and custom tokens -->

- ### Semantic Similarity sentence-transformers/all-MiniLM-L6-v2

    Onnx export of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

- ### FlowtronTTS with Waveglow vocoder

    Nvidia's [FlowtronTTS](https://github.com/NVIDIA/flowtron) architecture using Waveglow vocoder. 
    Weights were published by the authors, this model uses 
    [Flowtron LibriTTS2K](https://drive.google.com/file/d/1sKTImKkU0Cmlhjc_OeUDLrOLIXvUPwnO/view) version.

- ### Speech to text NeMo models

    This model is still heavy WIP it is best to use your platform's of choice existing solutions
    e.g. [UnityEngine.Windows.Speech.DictationRecognizer](https://docs.unity3d.com/ScriptReference/Windows.Speech.DictationRecognizer.html) in Unity.  

    This implementation uses several models exported from [NeMo](https://github.com/NVIDIA/NeMo) toolkit:

    - [QuartzNet15x5](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5) for transcription.
    - [Punctuation BERT](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/punctuation_en_bert) for applying punctuation.
    - Custom transformer for recognizing end of response to the context initialized from [all-MiniLM-L6-v2](nreimers/MiniLM-L6-H384-uncased)


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