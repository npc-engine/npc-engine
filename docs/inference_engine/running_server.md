
First lets get the `npc-engine`.

You can get it from

* [Releases page](https://github.com/eublefar/chatbot_server/releases)  
* Resulting folder after following [build instructions](../building/)

Inside the *npc-engine* folder *cli.exe* can be found. This is the CLI interface
to `npc-engine` server. 

You can check all the possible commands via:
```
cli.exe --help
```

To start the server create models directory:
```
mkdir models
```
and execute cli.exe with run command
```
cli.exe run --models-path models --port 5555
```
This will start a server but if no models were added to the folder it won't expose any API.

You can grab default models from [here](https://drive.google.com/drive/folders/1_3iOrhgvDyrKnC-tWEdysxpJyUcun0X3?usp=sharing).

!!! note "NOTE"
    Model API examples can be found in `npc-engine\tests\integration`.   
    If you don't need any specific model functionality just don't add this model to your *models* folder.

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