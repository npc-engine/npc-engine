from multiprocessing import context
import os
import inspect
import sys

import pytest
from npc_engine.services.persona_dialogue.persona_dialogue import PersonaDialogue

from npc_engine.services.utils.config import get_type_from_dict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from mocks.service_client_mocks import (
    MockControlClient,
    MockSimilarityClient,
    MockTextGenerationClient,
)
from mocks.zmq_mocks import Context


class TestPersonaDialog:
    def test_step_dialogue(self):
        PersonaDialogue.create_client = (
            lambda self, name: MockControlClient()
            if name == "control"
            else MockSimilarityClient()
            if name == "SimilarityAPI"
            else MockTextGenerationClient(
                context_template={
                    "persona": "",
                    "name": "",
                    "location": "",
                    "location_name": "",
                    "other_name": "",
                    "other_persona": "",
                    "history": [],
                }
            )
        )
        persona_dialogue = PersonaDialogue(
            service_id="test", uri="inproc://test", context=Context()
        )
        persona_dialogue.start_dialogue(
            name1="test_speaker1",
            persona1="test2",
            name2="test_speaker2",
            persona2="test4",
            location_name="test5",
            location_description="test6",
            dialogue_id="test7",
        )
        assert persona_dialogue.get_history("test7") == []
        persona_dialogue.step_dialogue("test7", "test_speaker1", "test1")
        assert persona_dialogue.get_history("test7") == [
            {"speaker": "test_speaker1", "line": "test1"}
        ]
        persona_dialogue.step_dialogue("test7", "test_speaker2", "test2")
        assert persona_dialogue.get_history("test7") == [
            {"speaker": "test_speaker1", "line": "test1"},
            {"speaker": "test_speaker2", "line": "test2"},
        ]
        persona_dialogue.end_dialogue("test7")

    def test_incorrect_context(self):
        PersonaDialogue.create_client = (
            lambda self, name: MockControlClient()
            if name == "control"
            else MockSimilarityClient()
            if name == "SimilarityAPI"
            else MockTextGenerationClient(
                context_template={
                    "persona": "",
                    "name": "",
                    "location": "",
                    "location_name": "",
                    "other_persona": "",
                    "history": [],
                }
            )
        )
        with pytest.raises(ValueError):
            persona_dialogue = PersonaDialogue(
                service_id="test", uri="inproc://test", context=Context()
            )
