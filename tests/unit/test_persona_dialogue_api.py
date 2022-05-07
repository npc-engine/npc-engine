from typing import Any, Dict, List
from npc_engine.services.persona_dialogue.persona_dialogue_base import (
    PersonaDialogueAPI,
)
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
import mocks.zmq_mocks as zmq


class MockPersonaDialogue(PersonaDialogueAPI):
    def __init__(self) -> None:
        super().__init__(context=zmq.Context(), service_id="test", uri="inproc://test")

    def generate_utterance(self, dialogue_id: str, speaker: int) -> str:
        return "test1"

    def start_dialogue(
        self,
        name1: str = None,
        persona1: str = None,
        name2: str = None,
        persona2: str = None,
        location_name: str = None,
        location_description: str = None,
        items_of_interest: List[str] = None,
        dialogue_id: str = None,
        other: Dict[str, Any] = None,
    ) -> str:
        return "test_dialogue_id"

    def update_dialogue(self, dialogue_id: str, speaker_id: int, utterance: str):
        assert dialogue_id == "test_dialogue_id"
        assert speaker_id == 0
        assert utterance == "test1"

    def end_dialogue(self, dialogue_id: str):
        assert dialogue_id == "test_dialogue_id"

    def check_scripted_utterances(
        self,
        utterance: str,
        scripted_utterances: List[str],
        scripted_threshold: float,
    ) -> bool:
        assert utterance == "test1"

    def get_history(self, dialogue_id: str) -> List[Dict[str, Any]]:
        pass


def test_persona_dialogue_api():

    api = MockPersonaDialogue()

    result = api.step_dialogue(
        "test_dialogue_id", 0, "test1", scripted_utterances=["test"]
    )
    assert ("test1", False) == result
    result = api.step_dialogue("test_dialogue_id", 0, scripted_utterances=["test"])
    assert ("test1", False) == result
