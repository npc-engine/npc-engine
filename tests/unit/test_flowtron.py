"""Speech synthesis test."""
import os
from npc_engine.models import Model
import time
from queue import Queue
import numpy as np
import sounddevice as sd


def test_flowtron():
    """Run flowtron inference, skip if no models in resources."""
    try:
        tts_module = Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..\\..\\npc_engine\\resources\\models\\flowtron",
            )
        )
    except FileNotFoundError:
        return
    start = time.time()

    test_line = "It's better sound realistic"
    start = time.time()
    audio = tts_module.run("6", test_line, 10)

    queue = Queue()

    def callback(indata, outdata, frames, time, status):
        if not queue.empty():
            arr = np.zeros((10240, 1))
            inp = queue.get(False)
            arr[: inp.shape[0], 0] = inp
            outdata[:] = arr

    stream = sd.Stream(
        channels=1, samplerate=22050, callback=callback, blocksize=10240
    ).__enter__()
    full_audio = []
    for i, audio_el in enumerate(audio):
        end = time.time()
        process_time = end - start
        audio_time = len(audio_el) / 22050
        if i == 0:
            audio_el[:1000] = 0
        queue.put(audio_el)
        full_audio += audio_el.tolist()
        print(f" > Step Processing time: {process_time}")
        print(f" > Step Real-time factor (should be < 1): {process_time / audio_time}")
        start = time.time()

    while not queue.empty():
        sd.sleep(int(10240 / 22.05))
    sd.sleep(int(10240 / 22.05))
    audio_time = len(full_audio) / 22050
