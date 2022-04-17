# """ZMQ server test."""
# import json
# import os
# from typing import Tuple
# from npc_engine.service_manager.server import Server
# from npc_engine.service_manager.service_manager import ServiceManager


# class StubSocket:
#     pass


# def test_zmq_json_rpc_server():
#     path = os.path.join(os.path.dirname(__file__), "..", "resources", "models")

#     server = Server("5555", ServiceManager(path))

#     print("Should reveive dummy")

#     async def recv_multipart(self):
#         return (
#             "HfChatbot",
#             json.dumps(
#                 {"jsonrpc": "2.0", "method": "dummy", "id": 0, "params": ["foo", "bar"]}
#             ),
#         )

#     async def send_multipart(self, tupl: Tuple[str, str]):
#         address, message = tupl
#         assert address == "HfChatbot"
#         message = json.loads(message)
#         assert message["id"] == 0
#         assert message["jsonrpc"] == "2.0"
#         assert "result" in message
#         raise KeyboardInterrupt()

#     StubSocket.send_multipart = send_multipart
#     StubSocket.recv_multipart = recv_multipart
#     stub = StubSocket()

#     server.socket = stub
#     api_dict = {
#         "dummy": lambda a, b: print(f" dummy API call with \na: {a}\nb: {b}"),
#         "dummy1": lambda a, b, c: print(
#             f" dummy1 API call with \na: {a}\nb: {b}\nc: {c}"
#         ),
#     }
#     try:
#         server.run()
#     except KeyboardInterrupt:
#         pass


# def test_zmq_json_rpc_server1():
#     path = os.path.join(os.path.dirname(__file__), "..", "resources", "models")

#     server = Server("5555", ServiceManager(path))

#     print("Should reveive dummy1")

#     def recv_multipart(self):
#         return (
#             "HfChatbot",
#             json.dumps(
#                 {
#                     "jsonrpc": "2.0",
#                     "method": "dummy1",
#                     "id": 0,
#                     "params": ["foo", "bar", "baz"],
#                 }
#             ),
#         )

#     async def send_multipart(self, tupl: Tuple[str, str]):
#         address, message = tupl
#         assert address == "HfChatbot"
#         message = json.loads(message)
#         assert message["id"] == 0
#         assert message["jsonrpc"] == "2.0"
#         assert "result" in message
#         raise KeyboardInterrupt()

#     StubSocket.send_multipart = send_multipart
#     StubSocket.recv_multipart = recv_multipart
#     stub = StubSocket()

#     server.socket = stub
#     api_dict = {
#         "dummy": lambda a, b: print(f" dummy API call with \na: {a}\nb: {b}"),
#         "dummy1": lambda a, b, c: print(
#             f" dummy1 API call with \na: {a}\nb: {b}\nc: {c}"
#         ),
#     }
#     try:
#         server.run()
#     except KeyboardInterrupt:
#         pass


# def test_zmq_json_rpc_server_incorrect_signature():
#     path = os.path.join(os.path.dirname(__file__), "..", "resources", "models")

#     server = Server("5556", ServiceManager(path))

#     def recv_multipart(self):
#         return (
#             "HfChatbot",
#             json.dumps(
#                 {
#                     "jsonrpc": "2.0",
#                     "method": "dummy",
#                     "id": 0,
#                     "params": ["foo", "bar", "baz"],
#                 }
#             ),
#         )

#     async def send_multipart(self, tupl: Tuple[str, str]):
#         address, message = tupl
#         assert address == "HfChatbot"
#         message = json.loads(message)
#         assert message["id"] == 0
#         assert message["jsonrpc"] == "2.0"
#         assert "result" in message
#         raise KeyboardInterrupt()

#     StubSocket.send_multipart = send_multipart
#     StubSocket.recv_multipart = recv_multipart
#     stub = StubSocket()
#     server.socket = stub

#     def incorrect_signature(foo):
#         print(f"foo {foo}")

#     api_dict = {"dummy": lambda *args, **kwargs: incorrect_signature(*args, **kwargs)}
#     try:
#         server.run(api_dict)
#     except KeyboardInterrupt:
#         pass
