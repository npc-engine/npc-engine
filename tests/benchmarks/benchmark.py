"""NPC engine integration tests"""
import subprocess
import os
import zmq
import time
import pytest
from loguru import logger


@pytest.mark.skip()
class TestClass:
    """Benchmars on the models in npc_engine/resources/models"""

    def setup_class(cls):
        logger.info("GPU VRAM Before starting inference engine:")
        os.system('nvidia-smi --query-gpu="memory.used" --format=csv')
        cli_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "npc_engine", "cli.py"
        )
        models_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "npc_engine", "resources", "models"
        )
        server_process = subprocess.Popen(
            ["python", cli_path, "run", "--models-path", models_path, "--port", "5555",]
        )
        cls.server_process = server_process
        time.sleep(25)  # Wait until server has loaded all the models
        logger.info("GPU VRAM After starting inference engine:")
        os.system('nvidia-smi --query-gpu="memory.used" --format=csv')

    def test_similarity_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "compare",
            "id": 0,
            "params": ["I will help you", ["I shall provide you my assistance"]],
        }
        start = time.time()
        socket.send_json(request)
        message = socket.recv_json()
        logger.info(f"Simple semantic similarity processed in {time.time() - start}s")
        assert "result" in message

    def test_tts_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        request = {
            "jsonrpc": "2.0",
            "method": "tts_start",
            "id": 0,
            "params": ["0", "I shall provide you my assistance", 10],
        }
        start = time.time()
        socket.send_json(request)
        message = socket.recv_json()
        request = {
            "jsonrpc": "2.0",
            "method": "tts_get_results",
            "id": 0,
            "params": [],
        }
        socket.send_json(request)
        message = socket.recv_json()
        logger.info(
            f"Text to speech latency (time before first result) {time.time() - start}"
        )
        # TODO: Add realtime factor for next results

    def test_chatbot_api(self):
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to npc-engine server")
        socket = context.socket(zmq.REQ)
        socket.RCVTIMEO = 2000
        socket.connect("tcp://localhost:5555")

        context = dict(
            persona="""
_setting_name Brimswood pub, Tavern
_setting_desc The Brimswood pub is an old establishment. 
It is sturdy, has a lot of life in its walls, but hasn't been updated in decades.
The clientele are the same as they always are, and they don't see very many strangers. 
The vibe is somber, and conversations are usually had in hushed tones.</s>
<speaker_self>_self_name pet dog
_self_persona I am mans best friend and I wouldn't have it any other way. I tend to my master and never leave his side. 
I sleep at his feet and guard the room at night from things that go bump in the night.</s>
<speaker_other>_partner_name the town baker's husband
_other_persona I am the town baker's husband and I love eating pastries.  
I tend to be in very good spirits and enjoy selling delicious baked goods that my wife has made.  
My wife is great at baking but she is lousy at washing my clothes.  
They keep shrinking!
        """.strip(),
            history=["<speaker_other>Hello friend!"],
        )

        request = {
            "jsonrpc": "2.0",
            "method": "generate_reply",
            "id": 0,
            "params": [context],
        }
        start = time.time()
        socket.send_json(request)
        message = socket.recv_json()
        logger.info(
            f"Chatbot reply {message['result']} processed in {time.time() - start}s"
        )
        assert "result" in message

    def teardown_class(cls):
        cls.server_process.terminate()
