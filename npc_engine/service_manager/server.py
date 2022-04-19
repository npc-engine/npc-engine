"""Module that implements ZMQ server communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
import json
import logging
import zmq
from loguru import logger
import time
import asyncio
import zmq  # noqa: F811
import zmq.asyncio
import traceback as tb

from npc_engine.service_manager.service_manager import ServiceManager


class Server:
    """Json rpc server over zmq."""

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        service_manager: ServiceManager,
        port: str,
    ):
        """Create a server on the port."""
        self.context = zmq_context
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.service_manager = service_manager

    def run(self):
        """Run an npc-engine json rpc server and start listening."""
        logger.info("Starting server")
        asyncio.get_event_loop().run_until_complete(self.loop())

    async def loop(self):
        """Run the server loop."""
        await asyncio.gather(self.msg_loop(), self.interrupt_loop())

    async def interrupt_loop(self):
        """Handle interrupts loop."""
        while True:
            await asyncio.sleep(1)

    async def msg_loop(self):
        """Asynchoriniously handle a request and reply."""
        logger.info("Starting message loop")
        while True:
            address = await self.socket.recv_string()
            _ = await self.socket.recv()
            message = await self.socket.recv_string()
            logger.info(f"Received request to {address}: {message}")
            asyncio.create_task(self.handle_reply(address, message))

    async def handle_reply(self, address: str, message: str):
        """Handle message and reply."""
        logging.info("Handling reply")
        start = time.time()
        try:
            response = await self.service_manager.handle_request(address, message)
        except Exception as e:
            response = {
                "code": -32000,
                "message": f"Internal error: {e}",
                "data": tb.extract_tb(e.__traceback__).format()
                if hasattr(e, "__traceback__")
                else None,
            }
            response = json.dumps(response)
        end = time.time()

        logger.info("Handle message time: %d" % (end - start))
        logger.info("Message reply: %s" % (response))

        #  Send reply back to client
        await self.socket.send_string(address, zmq.SNDMORE)
        await self.socket.send_string("", zmq.SNDMORE)
        await self.socket.send_string(response)
