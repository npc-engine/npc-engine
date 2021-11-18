"""Similarity test."""
import os
from npc_engine import models
import time
import pytest
import numpy
import scipy.signal
from pydub import AudioSegment


@pytest.mark.skip()
def test_sanity_check():
    """Check custom testing"""
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
    """Check custom testing"""
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
