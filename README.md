# npc-engine

NPC-Engine is a deep learning inference engine for designing NPC AI with natural language.

[![Build Status](https://github.com/npc-engine/npc-engine/actions/workflows/documentation_master.yml/badge.svg)](https://npc-engine.github.io/npc-engine/)
[![Build Status](https://github.com/npc-engine/npc-engine/actions/workflows/ci.yml/badge.svg)](https://npc-engine.github.io/npc-engine/)

## Features

- Chat-bot dialogue system.
- SoTA tools like text semantic similarity and text to speech.
- Easy, open source deep learning model standard (ONNX with YAML configs).
- GPU accelerated inference with onnxruntime.
- Engine agnostic API through ZMQ server via [JSONRPC 2.0](https://www.jsonrpc.org/specification).

## Getting started

The easiest way to get started is to use NPC Engine through the [Unity integration](https://assetstore.unity.com/packages/tools/ai/npc-engine-208498)

You can also use it via ZMQ library. See [Documentation](https://npc-engine.com/stable/inference_engine/running_server/) for more details.

## Roadmap

### Done:

- Real-time end-to-end chatbot dialogue system
- Semantic similarity
- Real-time speech to text (experimental)
- Unity integration
- CLI tool for importing models from [Huggingface](https://huggingface.co/transformers/index.html)
- Asynchronous API features

### In progress:

- Actions and planning
- Unreal integration
- Importing models from popular TTS libraries
- Emotion features
- Multiple languages support
- **Much more**

## Build on Windows

- Create virtualenv and activate it:

    ```cmd
    > python3 -m venv npc-engine-venv
    > .\npc-engine-venv\activate.bat
    ```

- Install dependencies

    ```
    > pip install -e .[dev,dml]
    ```

- (Optional) Compile, build and install your custom ONNX python runtime

    Build instructions here https://onnxruntime.ai/

- (Optional) Run tests

    + Download models to run tests against into `npc-engine\resources\models`.  
    ```
    npc-engine download-default-models --models-path npc-engine\resources\models
    ```
    + Run tests with
    ```
    > tox
    ```

- Compile to exe with:

    ```
    > pyinstaller --additional-hooks-dir hooks --exclude-module matplotlib --exclude-module jupyter --exclude-module torch --exclude-module torchvision .\npc-engine\cli.py --onedir
    ```

## Authors

- **eublefar** - _Python, Neural Nets_ - [github](https://github.com/eublefar)
- **igorzmitrovich** - _Python, CI/CD_ - [github](https://github.com/igorzmitrovich)

See also the list of [contributors](https://github.com/npc-engine/npc-engine/contributors) who participated in this project.
