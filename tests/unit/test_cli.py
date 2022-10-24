import os
import re
from npc_engine.cli import list_models, describe, set_models_path
from click.testing import CliRunner


def test_list_models():
    runner = CliRunner()
    test_models = """
mock-bart-light-gail-chatbot
BartChatbot
Service description:
BART based chatbot implementation class.
Model description:

# BART chatbot trained on [LIGHT](https://parl.ai/projects/light/) dataset with [Text Generative Adversarial Imitation Learning](https://arxiv.org/abs/2004.13796)
--------------------
mock-distilgpt2
HfChatbot
Service description:
Chatbot that uses Huggingface transformer architectures.
Model description:

--------------------
mock-distilroberta-base
HfClassifier
Service description:
Huggingface transformers sequence classification.
Model description:

--------------------
mock-flowtron-waveglow-librispeech-tts
FlowtronTTS
Service description:
Implements Flowtron architecture inference.
Model description:

# Exported [FlowtronTTS](https://arxiv.org/abs/2005.05957) with [WaveGlow](https://arxiv.org/abs/1811.00002) vocoder
--------------------
mock-paraphrase-MiniLM-L6-v2
TransformerSemanticSimilarity
Service description:
Huggingface transformers semantic similarity.
Model description:

# Export of [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
--------------------
    """
    result = runner.invoke(list_models, ["--models-path", "tests\\resources\\models"])
    if result.exit_code != 0:
        print("OUTPUT:", result.output, result.stdout)
        print(result.output)
    assert result.exit_code == 0
    assert result.output.strip() == test_models.strip()


def test_describe():
    runner = CliRunner()
    test_models = """
mock-distilgpt2

Service description:
Chatbot that uses Huggingface transformer architectures.

    ONNX export of Huggingface transformer is required (see https://huggingface.co/docs/transformers/serialization).
    Features seq2seq-lm, causal-lm, seq2seq-lm-with-past, causal-lm-with-past are supported
    
Model description:


    """
    result = runner.invoke(
        describe, ["mock-distilgpt2", "--models-path", "tests\\resources\\models"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == test_models.strip()


def test_set_models_path():
    runner = CliRunner()
    result = runner.invoke(
        set_models_path, ["--models-path", "tests\\resources\\models"]
    )
    assert result.exit_code == 0
    assert os.environ["NPC_ENGINE_MODELS_PATH"] == "tests\\resources\\models"
