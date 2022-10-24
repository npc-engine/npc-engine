# npc-engine

NPC-Engine is a deep learning and NLP toolkit for designing NPC AI with natural language.

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

You can also use it directly through ZMQ or HTTP. See [Documentation](https://npc-engine.com/stable/inference_engine/running_server/) for more details.

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
    ```
    > tox
    ```

- Compile to exe with:

    ```
    > pyinstaller --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" ^
    --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" ^
    --hidden-import="sklearn.neighbors._typedefs" --hidden-import="sklearn.utils._typedefs" ^
    --hidden-import="sklearn.neighbors._partition_nodes" --additional-hooks-dir hooks ^
    --exclude-module tkinter --exclude-module matplotlib .\npc_engine\cli.py --onedir
    ```

## Docker

If you wish to host NPC Engine somewhere you can use our the docker image. It's Linux image with TensorRT ONNX Runtime provider.

You can build it yourself with:

```bash
docker build -t npc-engine .
```

To run the image you must mount the models directory to `/app/models` e.g.

```bash
docker run --gpus all -it --mount type=bind,source=%cd%\tests\resources\models,target=/app/models -p 5000:5000 npc-engine/inference-engine:latest npc-engine run --port 5000
```

Where `--gpus all` will give access to the GPU, `-it` will output logs and let you use the container interactively, `--mount` will mount the models directory to the container, `-p 5000:5000` will expose the port 5000 on the host machine.


## Community

We have a [Discord](https://discord.gg/R4zBNmnfrU) server where you can get support, ask questions and show off your creations.

If you would like to donate, you can check out our [Patreon](https://www.patreon.com/npcengine).

### Our Patrons

- Marrech Games

## Authors

- **eublefar** - _Python, Neural Nets_ - [github](https://github.com/eublefar)
- **igorzmitrovich** - _Python, CI/CD_ - [github](https://github.com/igorzmitrovich)

See also the list of [contributors](https://github.com/npc-engine/npc-engine/contributors) who participated in this project.
