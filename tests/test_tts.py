"""
.. currentmodule:: test_example
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Speech synthesis test.
"""
import os
import simpleaudio as sa
from inference_engine.speech_synthesis import TTS
import time
from queue import Queue
import numpy as np
import sounddevice as sd


def test_reply_default():
    """Check if voice with no traits works
    """
    tts_module = TTS(
        os.path.join(
            os.path.dirname(__file__),
            "..\\inference_engine\\resources\\models\\flowtron_squeezewave",
        )
    )
    start = time.time()

    test_line = "It's better sound realistic"
    start = time.time()
    audio = tts_module.tts(6, test_line)

    queue = Queue()

    def callback(indata, outdata, frames, time, status):
        if not queue.empty():
            arr = np.zeros((5120, 1))
            inp = queue.get(False)
            arr[: inp.shape[0], 0] = inp
            outdata[:] = arr

    stream = sd.Stream(
        channels=1, samplerate=22050, callback=callback, blocksize=5120
    ).__enter__()
    full_audio = []
    start = time.time()
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
        sd.sleep(int(5120 / 22.05))
    end = time.time()
    process_time = end - start
    audio_time = len(full_audio) / 22050
    print(f" > Processing time: {process_time}")
    print(f" > Real-time factor: {process_time / audio_time}")

