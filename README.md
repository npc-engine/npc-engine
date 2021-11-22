# npc-engine

NPC-Engine is a deep learning inference engine for designing NPC AI with natural language.

[![Build Status](https://github.com/npc-engine/npc-engine/actions/workflows/Documentation.yml/badge.svg)](https://npc-engine.github.io/npc-engine/)
[![Build Status](https://github.com/npc-engine/npc-engine/actions/workflows/ci.yml/badge.svg)](https://npc-engine.github.io/npc-engine/)

## Features

- Chat-bot dialogue system.
- SoTA tools like text semantic similarity and text to speech.
- Easy, open source deep learning model standard (ONNX with YAML configs).
- GPU accelerated inference with onnxruntime.
- Engine agnostic API through ZMQ server via [JSONRPC 2.0](https://www.jsonrpc.org/specification).

## Getting started

First lets get the `npc-engine`.

It can be found in:
- [releases page](https://github.com/eublefar/chatbot_server/releases) and extract it to `npc-engine` folder
-  resulting folder after following [build instructions](#-build-on-windows)
-  one of the game engine integrations (coming soon).

Inside the *npc-engine* folder *cli.exe* can be found. This is the CLI interface
to `npc-engine` server. 

You can check all the possible commands via:
```
> cli.exe --help
```

To start the server create models directory:
```
> mkdir models
```
and execute cli.exe with run command
```
cli.exe run --models-path models --port 5555
```
This will start a server but if no models were added to the folder it won't expose any API.

You can grab default models from [here](https://drive.google.com/drive/folders/1_3iOrhgvDyrKnC-tWEdysxpJyUcun0X3?usp=sharing).

NOTE: Model API examples can be found in `npc-engine\tests\integration`. If you don't need any specific model functionality just don't add this models to your *models* folder.

Now lets test npc-engine with this example request from python:

```python
import zmq
context = zmq.Context()

#  Socket to talk to server
print("Connecting to npc-engine server")
socket = context.socket(zmq.REQ)
socket.RCVTIMEO = 2000
socket.connect("tcp://localhost:5555")
request = {
    "jsonrpc": "2.0",
    "method": "compare",
    "id": 0,
    "params": ["I will help you", ["I shall provide you my assistance"]],
}
socket.send_json(request)
message = socket.recv_json()
print(f"Response message {message}")
```

## Roadmap

### Done:

- Real-time end-to-end chatbot dialogue system
- Semantic similarity
- Real-time speech to text

### In progress:

- Unity integration
- Unreal integration
- CLI tool for importing models from [Huggingface](https://huggingface.co/transformers/index.html) and [Coqui TTS](https://tts.readthedocs.io/en/latest/)
- Asynchronous API features
- Emotion features
- Multiple languages support
- Behaviours from semantics (next action prediction)
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
    You can use default models from [here](https://drive.google.com/drive/folders/1_3iOrhgvDyrKnC-tWEdysxpJyUcun0X3?usp=sharing)
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
