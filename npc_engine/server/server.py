"""Module that implements ZMQ server communication over JSON-RPC 2.0 (https://www.jsonrpc.org/specification)."""
from abc import ABC, abstractmethod
import sys
import json
import logging
from loguru import logger
import time
import asyncio
import zmq
import zmq.asyncio
import traceback as tb

from npc_engine.server.control_service import ControlService
from npc_engine.server.metadata_manager import MetadataManager
from aiohttp import web


# TODO:
# - Add HTTP server to avoid problems with ZMQ socket on integration side


class BaseServer(ABC):
    """Base JSON RPC server."""

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        service_manager: ControlService,
        metadata: MetadataManager,
        start_services: bool = True,
    ):
        """Initialize the server.

        Args:
            zmq_context: The ZMQ context.
            service_manager: Control service that manages services.
            metadata: Metadata manager.
            start_services: Start services on initialization.
        """
        self.context = zmq_context
        self.socket_ipc = self.context.socket(zmq.ROUTER)
        self.socket_ipc.setsockopt(zmq.LINGER, 0)
        self.socket_ipc.bind(metadata.build_ipc_uri("self"))

        self.service_manager = service_manager
        self.start_services = start_services

    @abstractmethod
    def run(self):
        """Run the server."""
        pass

    async def msg_loop(self, socket: zmq.asyncio.Socket):
        """Asynchoriniously handle a request and reply."""
        while True:
            address = await socket.recv()
            _ = await socket.recv()
            message = await socket.recv_string()
            logger.info(f"Received request to {address}: {message}")
            asyncio.create_task(self.handle_reply(socket, address, message))

    async def interrupt_loop(self):
        """Handle interrupts loop."""
        while True:
            await asyncio.sleep(1)

    async def handle_reply(self, socket, address: str, message: str):
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
        await socket.send(address, zmq.SNDMORE)
        await socket.send_string("", zmq.SNDMORE)
        await socket.send_string(response)


class ZMQServer(BaseServer):
    """Json rpc server over zmq."""

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        service_manager: ControlService,
        metadata: MetadataManager,
        start_services: bool = True,
    ):
        """Create a server on the port."""
        super().__init__(zmq_context, service_manager, metadata, start_services)
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{metadata.port}")

    def run(self):
        """Run an npc-engine json rpc server and start listening."""
        logger.info("Starting server")
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.get_event_loop().run_until_complete(self.loop())

    async def loop(self):
        """Run the server loop."""
        try:
            logger.info("Starting services")
            if self.start_services:
                for service in self.service_manager.services:
                    self.service_manager.start_service(service)
            logger.info("Starting message loop")
            await asyncio.gather(
                self.msg_loop(self.socket),
                self.msg_loop(self.socket_ipc),
                self.interrupt_loop(),
            )
        except Exception as e:
            logger.error(f"Error in message loop: {e}")
            logger.error(tb.format_exc())
            raise e
        finally:
            logger.info("Closing message loop")
            self.socket.close()
            self.socket_ipc.close()
            self.context.destroy()


class HTTPServer(BaseServer):
    """HTTP server to handle requests."""

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        metadata: MetadataManager,
        service_manager: ControlService,
        start_services: bool = True,
    ):
        """Create a server on the port."""
        super().__init__(zmq_context, service_manager, metadata, start_services)
        self.app = web.Application()
        self.app.add_routes([web.get("/", self.handle_request)])
        self.app.add_routes([web.get("/{name}", self.handle_request)])
        self.app.add_routes([web.post("/", self.handle_request)])
        self.app.add_routes([web.post("/{name}", self.handle_request)])

    def run(self):
        """Run an npc-engine json rpc server and start listening."""
        if self.start_services:
            for service in self.service_manager.services:
                self.service_manager.start_service(service)
        logger.info("Starting server")
        web.run_app(self.init_and_get_app(), port=self.metadata.port)

    async def init_and_get_app(self):
        """Run the ipc loop and return aiohttp app."""
        logger.info("Starting services")
        if self.start_services:
            for service in self.service_manager.services:
                self.service_manager.start_service(service)
        logger.info("Starting message loop")
        asyncio.create_task(
            asyncio.gather(self.msg_loop(self.socket_ipc), self.interrupt_loop(),)
        )
        return self.app

    async def handle_request(self, request):
        """Handle request."""
        try:
            address = request.transport.match_info.get("name")
            message = await request.text()
            response = await self.service_manager.handle_request(address, message)
        except Exception as e:
            response = {
                "code": -32000,
                "message": f"Internal error: {type(e)} {e}",
                "data": tb.extract_tb(e.__traceback__).format()
                if hasattr(e, "__traceback__")
                else None,
            }
            response = json.dumps(response)
        return web.Response(text=response)