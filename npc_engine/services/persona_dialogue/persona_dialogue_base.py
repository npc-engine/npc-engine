"""Module that implements persona dialogue API."""
from typing import Any, Dict, List, Tuple

from abc import abstractmethod
from npc_engine.services.base_service import BaseService


class PersonaDialogueAPI(BaseService):
    """Abstract base class for persona dialogue models."""

    API_METHODS: List[str] = ["start_dialogue", "step_dialogue", "get_history"]

    def __init__(self, *args, **kwargs) -> None:
        """Empty initialization method for API to be similar to other model base classes."""
        super().__init__(*args, **kwargs)
        self.initialized = True

    @classmethod
    def get_api_name(cls) -> str:
        """Get the API name."""
        return "PersonaDialogueAPI"

    @abstractmethod
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
        """Start a dialogue between two characters.

        All arguments are supposed to be natural language descriptions.

        Args:
            name1: Name of the first character.
            persona1: Persona of the first character.
            name2: Name of the second character.
            persona2: Persona of the second character.
            location_name: Name of the place where dialogue happens.
            location_description: Description of the place where dialogue happens.
            items_of_interest: List of items of interest that could be mentioned in the dialogue.
            dialogue_id: ID of the dialogue. If None it will be named automatically.
            other: Other information that could be used to start the dialogue.
        Returns:
            Dialogue id.
        """
        pass

    @abstractmethod
    def end_dialogue(self, dialogue_id: str):
        """End a dialogue between two characters.

        Args:
            dialogue_id: ID of the dialogue.
        """
        pass

    def step_dialogue(
        self,
        dialogue_id: str,
        speaker_id: str,
        utterance: str = None,
        scripted_utterances: List[str] = None,
        scripted_threshold: float = 0.5,
        update_history: bool = True,
    ) -> Tuple[str, bool]:
        """Step a dialogue between two characters.

        Args:
            dialogue_id: ID of the dialogue.
            speaker: 0 for the first character, 1 for the second character.
            utterance: Natural language utterance. If None it will be generated.
            scripted_utterances: List of natural language utterances
                that will be matched against utterance.
            scripted_threshold: Threshold for matching scripted utterances.
            update_history: If True, the dialogue history will be updated.
        Returns:
            str: Next utterance.
            bool: scripted utterance triggered
        """
        if utterance is None:
            utterance = self.generate_utterance(dialogue_id, speaker_id)
        scripted = False
        if scripted_utterances is not None:
            idx = self.check_scripted_utterances(
                utterance, scripted_utterances, scripted_threshold
            )
            if idx is not None:
                utterance = scripted_utterances[idx]
                scripted = True
        if update_history:
            self.update_dialogue(dialogue_id, speaker_id, utterance)
        return utterance, scripted

    @abstractmethod
    def generate_utterance(self, dialogue_id: str, speaker_id: str) -> str:
        """Generate an utterance for the given speaker.

        Args:
            dialogue_id: ID of the dialogue.
            speaker: 0 for the first character, 1 for the second character.
        """
        pass

    @abstractmethod
    def check_scripted_utterances(
        self, utterance: str, scripted_utterances: List[str], threshold: float
    ) -> int:
        """Check if the given utterance is one of the scripted utterances.

        Args:
            utterance: Natural language utterance.
            scripted_utterances: Natural language utterances.
            threshold: [0,1] threshold for the similarity between the utterance and the scripted utterances.

        Returns:
            id of the utterance, None if the utterance is not one of the scripted utterances.
        """
        pass

    @abstractmethod
    def update_dialogue(self, dialogue_id: str, speaker_id: str, utterance: str):
        """Update dialogue state.

        Args:
            dialogue_id: ID of the dialogue.
            speaker_id: 0 for the first character, 1 for the second character.
            utterance: Natural language utterance.
        """
        pass

    @abstractmethod
    def get_history(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """Get the history of a dialogue.

        Args:
            dialogue_id: ID of the dialogue.
        """
        pass
