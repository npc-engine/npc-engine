Service classes are specific implementations of API classes. 
They define the loading process in `__init__` function and 
all the abstract methods required by the API class to function.

## How services are configured?

[MetadataManager](../reference/#npc_engine.server.metadata_manager.MetadataManager) Scans the models folder. For each discovered subfolder ServiceManager validates the `config.yml` and creates descriptors with metadata for each service. The mandatory field of `config.yml` is `type` (or `model_type`) that must contain correct service class that was discovered and registered by `BaseService` parent class. This service class will be instantiated with parsed dictionary as parameters on [ControlService.start_service](../reference/#npc_engine.server.control_service.ControlService.start_service) request to `control` service. 


## How is their API exposed?

When service is started [ControlService](../reference/#npc_engine.server.control_service.ControlService) starts a new process with [BaseService](../reference/#npc_engine.services.base_service.BaseService) message handling loop. [BaseService](../reference/#npc_engine.services.base_service.BaseService) handles ZMQ IPC requests to it's exposed functions from API's API_METHODS class variable to the main process, while main process handles routing requests to this service. 

## Existing service classes

:::npc_engine.services.sequence_classifier.hf_classifier.HfClassifier
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.text_generation.hf_text_generation.HfChatbot
    rendering:
      show_root_heading: true
      show_source: false

:::npc_engine.services.text_generation.bart.BartChatbot
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

    [BartChatbot](../reference/#npc_engine.services.text_generation.bart.BartChatbot)
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


## Creating new Services

You can use this dummy model example to create your own:

```python
from npc_engine.services.text_generation.text_generation_base import TextGenerationAPI

class EchoService(TextGenerationAPI):

    def __init__(self, model_path:str, *args, **kwargs):
        print("model is in {model_path}")
    
    def get_special_tokens(self):
        return {}

    def run(self, prompt, temperature=1, topk=None):
        return prompt
```

!!! note "Dont forget"
    Import new model to npc-engine.services so that it is discovered. 

### Using other services from your service

You can use other services from your service by using service clients.

They can be created from inside the service with `self.create_client(name)` where `name` is the name of the dependency.
These clients expose the same API as the dependency and can be just called from your service.