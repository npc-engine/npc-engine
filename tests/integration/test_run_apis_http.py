"""NPC engine integration tests"""
import subprocess
import os
import time
from npc_engine.service_clients import ControlClient
import zmq
import aiohttp
import pytest


class TestHTTPServer:
    """Test that starts npc-engine server and tests all the APIs"""

    def setup_class(cls):
        cli_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "npc_engine", "cli.py"
        )
        models_path = os.path.join(
            os.path.dirname(__file__), "..", "resources", "models"
        )
        server_process = subprocess.Popen(
            [
                "python",
                cli_path,
                "--verbose",
                "run",
                "--models-path",
                models_path,
                "--port",
                "5555",
                "--start-all",
                "--protocol",
                "http",
            ],
        )
        cls.server_process = server_process
        cls.context = zmq.Context()
        print("Starting server")
        cls.cc = ControlClient(cls.context)
        all_running = False
        services = [svc["id"] for svc in cls.cc.get_services_metadata()]
        while not all_running:
            time.sleep(1)
            all_running = True
            for service in services:
                status = cls.cc.get_service_status(service)
                all_running = all_running and (status == "running")
                if status == "error":
                    raise Exception("Server failed to start")
            if not all_running:
                print(f"Not all services are running. Waiting...")

    @pytest.mark.asyncio
    async def test_no_id_similarity_api(self):
        async with aiohttp.ClientSession() as session:
            #  Socket to talk to server
            request = {
                "jsonrpc": "2.0",
                "method": "cache",
                "id": 0,
                "params": ["I shall provide you my assistance"],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message
            request = {
                "jsonrpc": "2.0",
                "method": "compare",
                "id": 0,
                "params": ["I will help you", ["I shall provide you my assistance"]],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

    @pytest.mark.asyncio
    async def test_no_id_tts_api(self):

        async with aiohttp.ClientSession() as session:

            request = {
                "jsonrpc": "2.0",
                "method": "get_speaker_ids",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message
            speaker_id = message["result"][0]

            request = {
                "jsonrpc": "2.0",
                "method": "tts_start",
                "id": 0,
                "params": [speaker_id, "I shall provide you my assistance", 10],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

            request = {
                "jsonrpc": "2.0",
                "method": "tts_get_results",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

    @pytest.mark.asyncio
    async def test_no_id_chatbot_api(self):
        async with aiohttp.ClientSession() as session:

            request = {
                "jsonrpc": "2.0",
                "method": "get_context_template",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message
            ctx = message["result"]

            request = {
                "jsonrpc": "2.0",
                "method": "get_prompt_template",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

            request = {
                "jsonrpc": "2.0",
                "method": "generate_reply",
                "id": 0,
                "params": [ctx],
            }
            async with session.get(
                "http://localhost:5555", verify_ssl=False, json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

    @pytest.mark.asyncio
    async def test_similarity_api(self):
        async with aiohttp.ClientSession() as session:
            request = {
                "jsonrpc": "2.0",
                "method": "cache",
                "id": 0,
                "params": ["I shall provide you my assistance"],
            }
            async with session.get(
                "http://localhost:5555/mock-paraphrase-MiniLM-L6-v2", json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

            request = {
                "jsonrpc": "2.0",
                "method": "compare",
                "id": 0,
                "params": ["I will help you", ["I shall provide you my assistance"]],
            }
            async with session.get(
                "http://localhost:5555/mock-paraphrase-MiniLM-L6-v2", json=request
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

    @pytest.mark.asyncio
    async def test_tts_api(self):
        # mock-flowtron-waveglow-librispeech-tts
        async with aiohttp.ClientSession() as session:
            request = {
                "jsonrpc": "2.0",
                "method": "get_speaker_ids",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555/mock-flowtron-waveglow-librispeech-tts",
                json=request,
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message
            speaker_id = message["result"][0]

            request = {
                "jsonrpc": "2.0",
                "method": "tts_start",
                "id": 0,
                "params": [speaker_id, "I shall provide you my assistance", 10],
            }
            async with session.get(
                "http://localhost:5555/mock-flowtron-waveglow-librispeech-tts",
                json=request,
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

            request = {
                "jsonrpc": "2.0",
                "method": "tts_get_results",
                "id": 0,
                "params": [],
            }
            async with session.get(
                "http://localhost:5555/mock-flowtron-waveglow-librispeech-tts",
                json=request,
            ) as resp:
                message = await resp.json()
                print(message)
                assert "result" in message

    def teardown_class(cls):
        services = cls.cc.get_services_metadata()
        for service in services:
            cls.cc.stop_service(service["id"])
        cls.context.destroy()
        cls.server_process.kill()

