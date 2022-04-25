"""Module that implements ZMQ server communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
import zmq
from loguru import logger
import time
from jsonrpc import JSONRPCResponseManager, dispatcher


class ZMQServer:
    """Json rpc server over zmq."""

    def __init__(self, port: str):
        """Create a server on the port."""
        print("starting server")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def run(self, api_dict):
        """Run an npc-engine json rpc server and start listening.

        Args:
            api_dict: A Mapping from method names to callables that implement this method.
        """
        dispatcher.update(api_dict)
        while True:
            message = self.socket.recv_string()
            logger.info("Received request: %s" % message)

            start = time.time()
            response = JSONRPCResponseManager.handle(message, dispatcher)
            end = time.time()

            logger.info("Handle message time: %d" % (end - start))
            logger.info("Message reply: %s" % (response.json))

            #  Send reply back to client
            self.socket.send_string(response.json)
