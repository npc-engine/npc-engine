"""Module that implements control service."""
from multiprocessing import Process
import asyncio
import json
import zmq
import zmq.asyncio

from npc_engine import services
from npc_engine.server.metadata_manager import MetadataManager
from jsonrpc import JSONRPCResponseManager, Dispatcher
from loguru import logger


class ServiceState:
    """Enum for the state of the service."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    AWAITING = "awaiting"
    TIMEOUT = "timeout"
    ERROR = "error"


def set_logger(logger_):
    """Set the logger for the service."""
    global logger
    logger = logger_


def service_process(metadata: MetadataManager, service_id: str, logger) -> None:
    """Service subprocess function.

    Starts the service and runs it's loop.
    """
    set_logger(logger)
    context = zmq.Context()
    service = services.BaseService.create(
        context,
        metadata.services[service_id].path,
        metadata.services[service_id].uri,
        service_id,
    )
    service.loop()


class ControlService:
    """Service that manages other services and routes requests."""

    def __init__(
        self,
        zmq_context: zmq.asyncio.Context,
        metadata_manager: MetadataManager,
    ) -> None:
        """Initialize control service.

        Args:
            zmq_context: asyncio zmq context
            server_port: server port that will be passed to services for inter-communication
        """
        self.server_port = metadata_manager.port
        self.metadata = metadata_manager
        self.control_dispatcher = Dispatcher()
        self.control_dispatcher.update(
            {
                "get_services_metadata": self.metadata.get_services_metadata,
                "get_service_metadata": self.metadata.get_metadata,
                "get_service_status": self.get_service_status,
                "start_service": self.start_service,
                "stop_service": self.stop_service,
                "restart_service": self.restart_service,
                "check_dependency": self.check_dependency,
            }
        )
        self.zmq_context = zmq_context
        self.services = {
            service_id: {
                "process": None,
                "socket": None,
                "state": ServiceState.STOPPED,
                "in_queue": None,
                "out_queue": None,
                "dispatch_coroutine": None,
            }
            for service_id in self.metadata.services.keys()
        }

    def __del__(self):
        """Stop all services."""
        if hasattr(self, "services"):
            for service_id, service in self.services.items():
                if service["state"] == ServiceState.RUNNING:
                    try:
                        self.stop_service(service_id)
                    except Exception:
                        pass

    async def handle_request(self, address: str, request: str) -> str:
        """Parse request string and route request to correct service.

        Args:
            address (str): address of the service (either model name or class name)
            request (str): jsonRPC string

        Returns:
            str: jsonRPC response
        """
        request_dict = json.loads(request)
        service_id = self.metadata.resolve_service(address, request_dict["method"])
        self.check_service(service_id)
        if address == "control":
            logger.trace(f"Request from {address}\n Request: {request}")
        else:
            logger.info(f"Request from {address} ({service_id})\n Request: {request}")
        if service_id == "control":
            return JSONRPCResponseManager.handle(request, self.control_dispatcher).json
        else:
            if self.services[service_id]["state"] != ServiceState.RUNNING:
                raise ValueError(f"Service {service_id} is not running")
            else:
                self.services[service_id]["in_queue"].put_nowait(request)
                try:
                    result = await self.services[service_id]["out_queue"].get()
                except asyncio.CancelledError:
                    result = ""
                return result

    def check_service(self, service_id):
        """Check if the service process is running."""
        if (
            service_id != "control"
            and (
                self.services[service_id]["state"] == ServiceState.RUNNING
                or self.services[service_id]["state"] == ServiceState.STARTING
                or self.services[service_id]["state"] == ServiceState.AWAITING
            )
            and not self.services[service_id]["process"].is_alive()
        ):
            self.services[service_id]["state"] = ServiceState.ERROR
            raise ValueError(f"Error in service {service_id}. Process is not alive.")

    def get_service_status(self, service_id):
        """Get the status of the service."""
        service_id = self.metadata.resolve_service(service_id, None)
        self.check_service(service_id)
        return self.services[service_id]["state"]

    def start_service(self, service_id):
        """Start the service."""
        service_id = self.metadata.resolve_service(service_id, None)
        self.check_service(service_id)
        if (
            self.services[service_id]["state"] == ServiceState.RUNNING
            or self.services[service_id]["state"] == ServiceState.STARTING
        ):
            raise ValueError(f"Service {service_id} is already running")

        process = Process(
            target=service_process,
            args=(self.metadata, service_id, logger),
            daemon=True,
        )
        process.start()
        self.services[service_id]["process"] = process
        self.services[service_id]["state"] = ServiceState.STARTING
        self.services[service_id]["socket"] = self.zmq_context.socket(zmq.REQ)
        self.services[service_id]["socket"].setsockopt(zmq.LINGER, 0)
        self.services[service_id]["socket"].setsockopt(zmq.RCVTIMEO, 10000)
        self.services[service_id]["socket"].connect(
            self.metadata.services[service_id].uri
        )
        self.services[service_id]["in_queue"] = asyncio.Queue()
        self.services[service_id]["out_queue"] = asyncio.Queue()
        loop = asyncio.get_event_loop()
        loop.create_task(
            self.confirm_state_coroutine(service_id),
            name=f"confirm_state_coroutine_{service_id}",
        )

    async def confirm_state_coroutine(self, service_id):
        """Confirm the state of the service."""
        request = json.dumps({"jsonrpc": "2.0", "method": "status", "id": 1})
        socket = self.services[service_id]["socket"]
        logger.info(f"Confirming state of {service_id}")
        try:
            await socket.send_string(request)
        except zmq.Again:
            self.services[service_id]["state"] = ServiceState.STARTING
            logger.warning(f"{service_id} is not responding.")
            await asyncio.sleep(1)
            await self.confirm_state_coroutine(service_id)
            return

        logger.info(f"Waiting for state of {service_id}")
        response = None
        while response is None:
            try:
                response = await socket.recv_string()
            except zmq.Again:
                self.services[service_id]["state"] = ServiceState.STARTING
                logger.warning(f"{service_id} is not responding.")
                await asyncio.sleep(1)
                continue

        resp_dict = json.loads(response)
        if resp_dict["result"] == ServiceState.RUNNING:
            logger.info(f"{service_id} is running")
            self.services[service_id]["state"] = ServiceState.RUNNING
            self.services[service_id]["dispatch_coroutine"] = asyncio.create_task(
                self.message_dispatch_coroutine(service_id),
                name=f"confirm_state_coroutine_{service_id}",
            )
        elif resp_dict["result"] == ServiceState.STARTING:
            logger.info(f"Service {service_id} responds but still starting")
            await asyncio.sleep(1)
            await self.confirm_state_coroutine(service_id)
        else:
            logger.warning(
                f"Service {service_id} failed to start and returned incorrect state."
            )
            self.services[service_id]["state"] = ServiceState.ERROR

    async def message_dispatch_coroutine(self, service_id: str):
        """Dispatch messages from the service."""
        try:
            while self.services[service_id]["state"] == ServiceState.RUNNING:
                request = await self.services[service_id]["in_queue"].get()
                logger.info(f"Dispatching message {request} to {service_id}")
                await self.services[service_id]["socket"].send_string(request)
                response = await self.services[service_id]["socket"].recv_string()
                logger.info(f"Received response {response} from {service_id}")
                self.services[service_id]["out_queue"].put_nowait(response)
        except RuntimeError as e:
            if e.args[0] == "Event loop is closed":
                logger.info(f"{service_id} is shutting down")
                return
            else:
                logger.error(f"{service_id} is shutting down. Exception: {e}")
                raise e

    def stop_service(self, service_id):
        """Stop the service."""
        service_id = self.metadata.resolve_service(service_id, None)
        self.check_service(service_id)
        if self.services[service_id]["state"] != ServiceState.RUNNING:
            raise ValueError(f"Service {service_id} is not running")
        self.services[service_id]["socket"].close()
        self.services[service_id]["socket"] = None
        self.services[service_id]["process"].terminate()
        self.services[service_id]["process"] = None
        self.services[service_id]["state"] = ServiceState.STOPPED
        self.services[service_id]["dispatch_coroutine"].cancel()
        self.services[service_id]["dispatch_coroutine"] = None

    def restart_service(self, service_id):
        """Restart the service."""
        self.stop_service(service_id)
        self.start_service(service_id)

    def check_dependency(self, service_id, dependency):
        """Check if the service has the dependency."""
        service_id = self.metadata.resolve_service(service_id, None)
        self.metadata.services[service_id].dependencies.append(dependency)
        self.metadata.check_dependency_cycles()
