"""Speech synthesis test."""
import os
from npc_engine.models import Model
import time
from queue import Queue
import numpy as np
import sounddevice as sd
import pytest


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "npc_engine",
            "resources",
            "models",
            "flowtron",
            "config.yml",
        )
    ),
    reason="Model missing",
)
def test_flowtron():
    """Run flowtron inference, skip if no models in resources."""
    try:
        tts_module = Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "npc_engine",
                "resources",
                "models",
                "flowtron",
            )
        )
    except FileNotFoundError:
        return
    start = time.time()

    test_line = "It should sound realistic. If it doesn't sound realistic, it's not real. But what is real? "
    start = time.time()
    tts_module.tts_start("6", test_line, 7)

    queue = Queue()

    def callback(indata, outdata, frames, time, status):
        if not queue.empty():
            arr = np.zeros((10240, 1))
            for i in range(10240):
                try:
                    arr[i, 0] = queue.get(False)
                except:
                    continue
            outdata[:] = arr

    play_audio = True
    try:
        stream = sd.Stream(
            channels=1, samplerate=22050, callback=callback, blocksize=10240
        ).__enter__()
    except sd.PortAudioError:
        play_audio = False

    full_audio = []
    i = -1
    while True:
        try:
            audio_el = np.asarray(tts_module.tts_get_results())
            i += 1
        except StopIteration:
            break
        end = time.time()
        process_time = end - start
        audio_time = len(audio_el) / 22050
        if i == 0:
            audio_el[:1000] = 0
        if play_audio:
            for j in range(audio_el.shape[0]):
                queue.put(audio_el[j])
        full_audio += audio_el.tolist()
        print(f" > Step Processing time: {process_time}")
        print(f" > Step Real-time factor (should be < 1): {process_time / audio_time}")
        start = time.time()
    if play_audio:
        while not queue.empty():
            sd.sleep(int(10240 / 22.05))
        sd.sleep(int(10240 / 22.05))
        stream.close()
