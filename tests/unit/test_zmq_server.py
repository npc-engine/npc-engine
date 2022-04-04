"""ZMQ server test."""
import json
from npc_engine.rpc.server import Server


class StubSocket:
    pass


def test_zmq_json_rpc_server():
    server = Server("5555")

    print("Should reveive dummy")

    def recv_string(self):
        return json.dumps(
            {"jsonrpc": "2.0", "method": "dummy", "id": 0, "params": ["foo", "bar"]}
        )

    def send_string(self, string: str):
        message = json.loads(string)
        assert message["id"] == 0
        assert message["jsonrpc"] == "2.0"
        assert "result" in message
        raise KeyboardInterrupt()

    StubSocket.send_string = send_string
    StubSocket.recv_string = recv_string
    stub = StubSocket()

    server.socket = stub
    api_dict = {
        "dummy": lambda a, b: print(f" dummy API call with \na: {a}\nb: {b}"),
        "dummy1": lambda a, b, c: print(
            f" dummy1 API call with \na: {a}\nb: {b}\nc: {c}"
        ),
    }
    try:
        server.run(api_dict)
    except KeyboardInterrupt:
        pass


def test_zmq_json_rpc_server1():
    server = Server("5555")

    print("Should reveive dummy1")

    def recv_string(self):
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "dummy1",
                "id": 0,
                "params": ["foo", "bar", "baz"],
            }
        )

    def send_string(self, string: str):
        message = json.loads(string)
        assert message["id"] == 0
        assert message["jsonrpc"] == "2.0"
        assert "result" in message
        raise KeyboardInterrupt()

    StubSocket.send_string = send_string
    StubSocket.recv_string = recv_string
    stub = StubSocket()

    server.socket = stub
    api_dict = {
        "dummy": lambda a, b: print(f" dummy API call with \na: {a}\nb: {b}"),
        "dummy1": lambda a, b, c: print(
            f" dummy1 API call with \na: {a}\nb: {b}\nc: {c}"
        ),
    }
    try:
        server.run(api_dict)
    except KeyboardInterrupt:
        pass


def test_zmq_json_rpc_server_incorrect_signature():
    server = Server("5555")

    def recv_string(self):
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "dummy",
                "id": 0,
                "params": ["foo", "bar", "baz"],
            }
        )

    def send_string(self, string: str):
        message = json.loads(string)
        assert message["id"] == 0
        assert message["jsonrpc"] == "2.0"
        assert "error" in message
        raise KeyboardInterrupt()

    StubSocket.send_string = send_string
    StubSocket.recv_string = recv_string
    stub = StubSocket()
    server.socket = stub

    def incorrect_signature(foo):
        print(f"foo {foo}")

    api_dict = {"dummy": lambda *args, **kwargs: incorrect_signature(*args, **kwargs)}
    try:
        server.run(api_dict)
    except KeyboardInterrupt:
        pass
