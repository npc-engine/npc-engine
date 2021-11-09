"""NPC engine integration tests"""
import subprocess
import os
import zmq
import time


class TestClass:
    """Test that starts npc-engine server and tests all the APIs"""

    def setup_class(cls):
        cli_path = os.path.join(os.path.dirname(__file__), "..\\..\\npc_engine\\cli.py")
        models_path = os.path.join(
            os.path.dirname(__file__), "..\\..\\npc_engine\\resources\\models"
        )
        server_process = subprocess.Popen(
            ["python", cli_path, "run", "--models-path", models_path, "--port", "5555",]
        )
        cls.server_process = server_process
        print("Starting server")
        time.sleep(25)  # Wait until server has loaded all the models

    def test_similarity_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "cache",
            "id": 0,
            "params": ["I shall provide you my assistance"],
        }

        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

        request = {
            "jsonrpc": "2.0",
            "method": "compare",
            "id": 0,
            "params": ["I will help you", ["I shall provide you my assistance"]],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

    def test_tts_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "get_speaker_ids",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message
        speaker_id = message["result"][0]

        request = {
            "jsonrpc": "2.0",
            "method": "tts_start",
            "id": 0,
            "params": [speaker_id, "I shall provide you my assistance", 10],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

        request = {
            "jsonrpc": "2.0",
            "method": "tts_get_results",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

    def test_chatbot_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "get_context_fields",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message
        ctx = message["result"]

        request = {
            "jsonrpc": "2.0",
            "method": "get_prompt_template",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

        request = {
            "jsonrpc": "2.0",
            "method": "generate_reply",
            "id": 0,
            "params": [ctx],
        }
        socket.send_json(request)
        message = socket.recv_json()
        assert "result" in message

    def teardown_class(cls):
        cls.server_process.terminate()
