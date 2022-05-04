"""NPC engine integration tests"""
import subprocess
import os
import zmq
import time
from npc_engine.service_clients import (
    ControlClient,
    ChatbotClient,
    SimilarityClient,
    SequenceClassifierClient,
)


class TestZMQServer:
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

    def test_no_id_similarity_api(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = type(self).context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
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
        print(message)
        assert "result" in message

    def test_no_id_tts_api(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = type(self).context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
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
        print(message)
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
        print(message)
        assert "result" in message

        request = {
            "jsonrpc": "2.0",
            "method": "tts_get_results",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        print(message)
        assert "result" in message

    def test_no_id_chatbot_api(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = type(self).context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "get_context_template",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        print(message)
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
        print(message)
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

    def test_similarity_api(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = type(self).context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.RCVTIMEO = 2000
        socket.setsockopt(zmq.IDENTITY, "mock-paraphrase-MiniLM-L6-v2".encode("utf-8"))
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
        print(message)
        assert "result" in message

    def test_tts_api(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = type(self).context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.RCVTIMEO = 2000
        socket.setsockopt(
            zmq.IDENTITY, "mock-flowtron-waveglow-librispeech-tts".encode("utf-8")
        )
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "get_speaker_ids",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        print(message)
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
        print(message)
        assert "result" in message

        request = {
            "jsonrpc": "2.0",
            "method": "tts_get_results",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        print(message)
        assert "result" in message

    def test_chatbot_api_self_client(self):

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        hf_chatbot = ChatbotClient(type(self).context)

        ctx = hf_chatbot.get_context_template()

        template = hf_chatbot.get_prompt_template()
        assert isinstance(template, str)

        reply = hf_chatbot.generate_reply(ctx)
        assert isinstance(reply, str)
        assert reply != ""

    def test_similarity_api_self_client(self):
        print("Connecting to npc-engine server")
        similarity_client = SimilarityClient(type(self).context)

        reply = similarity_client.compare(
            "I shall provide you my assistance", ["I shall provide you my assistance"]
        )

        assert isinstance(reply, list)
        assert isinstance(reply[0], float)

    def test_sequence_classifier_self_client(self):
        print("Connecting to npc-engine server")
        sequence_classifier_client = SequenceClassifierClient(type(self).context)

        reply = sequence_classifier_client.classify(
            ["I shall provide you my assistance", ["hello", "world"]]
        )

        assert len(reply) == 2
        assert len(reply[0]) == len(reply[1])

    def teardown_class(cls):
        services = cls.cc.get_services_metadata()
        for service in services:
            cls.cc.stop_service(service["id"])
        cls.context.destroy()
        cls.server_process.kill()
