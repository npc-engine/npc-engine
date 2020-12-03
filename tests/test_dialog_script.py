"""
.. currentmodule:: test_dialog_script
.. moduleauthor:: evil.unicorn1 <evil.unicorn1@gmail.com>

Dialog scripted test.
"""
import os
import simpleaudio as sa
from chatbot_server.dialog_script import DialogScriptSystem
import time
import logging


def test_simple_cue():
    """Check if dialog script is triggered by similar line
    """
    dialog_script = DialogScriptSystem(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\roberta_semb')
    )
    dialog_script.add_speaker("Barman")
    dialog_script.script_line(
        "Barman",
        "root",
        "buy_beer",
        ["Can I have a beer", "Give me a beer"],
        ["Sure, I'll get it in a moment", "Here, please", "In a moment"],
        0,
        0.6
    )
    start = time.time()
    response = dialog_script.step_dialog("Barman", "Can I have a beer")
    end = time.time()
    print(str(end - start) + ' seconds elapsed')
    assert response is not None
    assert dialog_script.current_nodes["Barman"] == ["root"]


def test_input_dissimilar_to_cue():
    """Check if chatbot works
    """
    dialog_script = DialogScriptSystem(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\roberta_semb')
    )
    dialog_script.add_speaker("Barman")
    dialog_script.script_line(
        "Barman",
        "root",
        "buy_beer",
        ["Can I have a beer", "Give me a beer"],
        ["Sure, I'll get it in a moment", "Here, please", "In a moment"],
        0,
        0.6
    )
    start = time.time()
    response = dialog_script.step_dialog("Barman", "How is it going")
    end = time.time()
    print(str(end - start) + ' seconds elapsed')
    assert response is None


def test_dialog_tree():
    """Check if dialog tree works
    """
    dialog_script = DialogScriptSystem(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\roberta_semb')
    )
    dialog_script.add_speaker("Barman")
    dialog_script.script_line(
        "Barman",
        "root",
        "ask_job",
        ["What do you need help with"],
        ["I need you to get me 5 pelts"],
        2,
        0.5
    )
    dialog_script.script_line(
        "Barman",
        "ask_job",
        "ask_job_info",
        ["What kind of pelts do you need"],
        ["I need wolf pelts"],
        1,
        0.6
    )
    response = dialog_script.step_dialog("Barman", "You needed help with something")
    assert response is not None
    assert "ask_job" in dialog_script.current_nodes["Barman"]
    response = dialog_script.step_dialog("Barman", "What pelts")
    assert response is not None
    assert dialog_script.current_nodes["Barman"] == ["root"]


def test_expiration():
    """Check if dialog tree nodes expire and return to root
    """
    dialog_script = DialogScriptSystem(
        os.path.join(os.path.dirname(__file__), '..\\chatbot_server\\resources\\models\\roberta_semb')
    )
    dialog_script.add_speaker("Barman")
    dialog_script.script_line(
        "Barman",
        "root",
        "ask_job",
        ["What do you need help with"],
        ["I need you to get me 5 pelts"],
        2,
        0.5
    )
    dialog_script.script_line(
        "Barman",
        "ask_job",
        "ask_job_info",
        ["What kind of pelts do you need"],
        ["I need wolf pelts"],
        1,
        0.6
    )
    response = dialog_script.step_dialog("Barman", "You needed help with something")
    assert response is not None
    assert "ask_job" in dialog_script.current_nodes["Barman"]

    response = dialog_script.step_dialog("Barman", "How do you like the weather")
    assert response is None
    assert "ask_job" in dialog_script.current_nodes["Barman"]

    response = dialog_script.step_dialog("Barman", "How do you like the weather")
    assert response is None
    assert dialog_script.current_nodes["Barman"] == ["root"]
    




