"""Dialog scripted test."""
import json
from inference_engine.zmq_server import ZMQServer


class StubSocket:
    def recv_string(self):
        return json.dumps(
            {"jsonrpc": "2.0", "method": "dummy", "id": 0, "params": ["foo", "bar"]}
        )


def test_zmq_json_rpc_server():
    server = ZMQServer("5555")

    def send_string(self, string: str):
        message = json.loads(string)
        assert message["id"] == 0
        assert message["jsonrpc"] == "2.0"
        assert "result" in message
        raise KeyboardInterrupt()

    StubSocket.send_string = send_string
    stub = StubSocket()

    server.socket = stub
    api_dict = {
        "dummy": lambda *args, **kwargs: print(
            f" dummy API call with \nargs: {args}\nkwargs: {kwargs}"
        )
    }
    try:
        server.run(api_dict)
    except KeyboardInterrupt:
        pass


def test_zmq_json_rpc_server_incorrect_signature():
    server = ZMQServer("5555")

    def send_string(self, string: str):
        message = json.loads(string)
        assert message["id"] == 0
        assert message["jsonrpc"] == "2.0"
        assert "error" in message
        raise KeyboardInterrupt()

    StubSocket.send_string = send_string
    stub = StubSocket()
    server.socket = stub

    def incorrect_signature(foo):
        print(f"foo {foo}")

    api_dict = {"dummy": lambda *args, **kwargs: incorrect_signature(*args, **kwargs)}
    try:
        server.run(api_dict)
    except KeyboardInterrupt:
        pass

