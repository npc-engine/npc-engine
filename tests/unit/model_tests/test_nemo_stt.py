"""Similarity test."""
import os
from npc_engine import models
import time
import pytest
import numpy
import scipy.signal
from pydub import AudioSegment
import numpy as np
import sounddevice as sd


@pytest.mark.skip()
def test_sanity_check():
    try:
        stt = models.Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "npc_engine",
                "resources",
                "models",
                "stt",
            )
        )
    except FileNotFoundError:
        return
    device = input(f"Select device: \n {stt.get_devices()} \n")
    stt.select_device(device)
    print("Listening...")
    result = stt.listen("hello how are you")
    print(f"Result: {result}")


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
            "stt",
            "config.yml",
        )
    ),
    reason="Model missing",
)
def test_transcribe():
    try:
        stt = models.Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "npc_engine",
                "resources",
                "models",
                "stt",
            )
        )
    except FileNotFoundError:
        return

    audio = AudioSegment.from_file(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "Recording (2).m4a",
        )
    )
    audio = numpy.frombuffer(audio.raw_data, numpy.int16)
    s = scipy.signal.decimate(audio, 6)
    s = s / 32767
    print(s.max())
    print(s.min())
    signal = s.astype(numpy.float32)

    start_trs = time.time()
    result = stt.transcribe(signal)
    assert result == "hello how is it going"
    end_trs = time.time()
    result = stt.postprocess(result)
    print(
        f"Result: {result} with transcription in {end_trs - start_trs} and postprocess in {time.time() - end_trs}"
    )
    assert result == "Hello, how is it going?"


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
            "stt",
            "config.yml",
        )
    ),
    reason="Model missing",
)
def test_transcribe_frame():
    try:
        stt = models.Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "npc_engine",
                "resources",
                "models",
                "stt",
            )
        )
    except FileNotFoundError:
        return

    audio = AudioSegment.from_file(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "Recording (2).m4a",
        )
    )
    audio = numpy.frombuffer(audio.raw_data, numpy.int16)
    s = scipy.signal.decimate(audio, 6)
    s = s / 32767
    print(f"Model frame_size {stt.frame_size}")
    signal = s.astype(numpy.float32)
    to_pad = stt.frame_size - signal.size % stt.frame_size
    signal = np.pad(signal, [(0, to_pad)], mode="constant", constant_values=0)
    signal = signal.reshape([-1, stt.frame_size])
    total_result = ""
    for frame in signal:
        start_trs = time.time()
        result = stt.transcribe_frame(frame)
        end_trs = time.time()
        total_result += result
        print(f"Result: {result} with transcription in {end_trs - start_trs}")
    sd.play(stt.buffer, samplerate=16000)
    result = stt.postprocess(total_result)
    print(f"End Result: {result}")
    assert result == "Hello, how is it going?"


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
            "stt",
            "config.yml",
        )
    ),
    reason="Model missing",
)
def test_decide_finished():
    try:
        stt = models.Model.load(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "npc_engine",
                "resources",
                "models",
                "stt",
            )
        )
    except FileNotFoundError:
        return

    assert stt.decide_finished("how do you feel", "i feel fine", 600)
    assert not stt.decide_finished("how do you feel", "i feel", 600)
