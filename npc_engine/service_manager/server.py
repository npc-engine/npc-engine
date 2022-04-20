"""Module that implements ZMQ server communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
import sys
import json
import logging
from loguru import logger
import time
import asyncio
import zmq
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
        start_services: bool = True,
    ):
        """Create a server on the port."""
        self.context = zmq_context
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.service_manager = service_manager
        self.start_services = start_services

    def run(self):
        """Run an npc-engine json rpc server and start listening."""
        logger.info("Starting server")
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
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
        logger.info("Starting services")
        if self.start_services:
            for service in self.service_manager.services:
                self.service_manager.start_service(service)
        logger.info("Starting message loop")
        while True:
            address = await self.socket.recv()
            _ = await self.socket.recv()
            message = await self.socket.recv_string()
            logger.info(f"Received request to {address}: {message}")
            asyncio.create_task(self.handle_reply(address, message))

    async def handle_reply(self, address: str, message: str):
        """Handle message and reply."""
        logging.info("Handling reply")
        start = time.time()
        try:
            address_str = address.decode("utf-8")
        except UnicodeDecodeError:
            address_str = address.hex()
        try:
            response = await self.service_manager.handle_request(address_str, message)
        except Exception as e:
            response = {
                "code": -32000,
                "message": f"Internal error: {type(e)} {e}",
                "data": tb.extract_tb(e.__traceback__).format()
                if hasattr(e, "__traceback__")
                else None,
            }
            response = json.dumps(response)
        end = time.time()

        logger.info("Handle message time: %d" % (end - start))
        logger.info("Message reply: %s" % (response))

        #  Send reply back to client
        await self.socket.send(address, zmq.SNDMORE)
        await self.socket.send_string("", zmq.SNDMORE)
        await self.socket.send_string(response)
