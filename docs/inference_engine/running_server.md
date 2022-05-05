
First lets get the `npc-engine`.

The simplest way to install npc-engine is to use the `pip` command.

```bash
pip install npc-engine[dml]
```

If you are running it on linux you should specify `cpu` extra instead of `dml` as DirectML works only on Windows.

To be able to ship it with your game you will need pyinstaller packaged version from:

* [Releases page](https://github.com/npc-engine/npc-engine/releases)  
* Resulting folder after following [build instructions](../building/)

If using packaged version you should run `cli.exe` inside the `npc-engine` folder instead of `npc-engine` command. 

You can check all the possible commands via:
```
npc-engine --help
```

To start the server create models directory:
```
mkdir models
```
and execute cli.exe with run command
```
npc-engine run --models-path models --port 5555
```
This will start a server but if no models were added to the folder it will expose only conrol API.

You can download default models via
```
npc-engine download-default-models --models-path models
```

See descriptions of the default models in [Default Models](../models/#default-models) section.

!!! note "NOTE"
    Model API examples can be found in `npc-engine\tests\integration`.   

Now lets test npc-engine with this example request from python:

- First start the server on port 5555:

- Now run the following python script:


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
    "method": "start_service",
    "id": 0,
    "params": ["SimilarityAPI"],
}
socket.send_json(request)
message = socket.recv_json()
request = {
    "jsonrpc": "2.0",
    "method": "compare",
    "id": 0,
    "params": ["I will help you", ["I shall provide you my assistance"]],
}
socket.send_json(request)
message = socket.recv_json()
print(f"Response message {message}")

# You can also provide socket identity to select a specific service

socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.IDENTITY, b"control")
socket.RCVTIMEO = 2000
socket.connect("tcp://localhost:5555")

request = {
    "jsonrpc": "2.0",
    "method": "get_services_metadata",
    "id": 0,
    "params": [],
}
socket.send_json(request)
message = socket.recv_json()
print(f"Services metadata {message}")
```