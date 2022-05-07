"""Persona Dialogue implementation using the provided services."""
from copy import copy
from typing import Any, Dict, List
from .persona_dialogue_base import PersonaDialogueAPI  # noqa: F401


class PersonaDialogue(PersonaDialogueAPI):
    """Persona dialogue API implementation using the provided services.

    Scripted utterances are matched via semantic similarity.
    Utterances are generated using the provided text generation service.
    """

    def __init__(
        self,
        text_generation_svc: str = "TextGenerationAPI",
        similarity_svc: str = "SimilarityAPI",
        *args,
        **kwargs,
    ):
        """Initialize the persona dialogue API.

        Args:
            text_generation_service: Name of the text generation service.
            similarity_api: Name of the similarity API.
            similarity_threshold: Threshold for semantic similarity.
        """
        super().__init__(*args, **kwargs)
        self.text_generation_service = self.create_client(text_generation_svc)
        self.similarity_api = self.create_client(similarity_svc)
        self.dialogues = {}
        self.dialogue_id_counter = 0
        self.context_template = self.text_generation_service.get_context_template()
        self._validate_context_template(self.context_template)

    def start_dialogue(
        self,
        name1: str = None,
        persona1: str = None,
        name2: str = None,
        persona2: str = None,
        location_name: str = None,
        location_description: str = None,
        dialogue_id: str = None,
        *args,
        **kwargs,
    ) -> str:
        """Start a dialogue between two characters.

        Args:
            dialogue_id: ID of the dialogue. If None it will be named automatically.
            name1: Name of the first character.
            persona1: Persona of the first character.
            name2: Name of the second character.
            persona2: Persona of the second character.
            location_name: Name of the place where dialogue happens.
            location_description: Description of the place where dialogue happens.

        Returns:
            Dialogue id.
        """
        if dialogue_id in self.dialogues:
            raise ValueError("Dialogue already exists.")
        if dialogue_id is None:
            dialogue_id = f"dialogue_{self.dialogue_id_counter}"
            self.dialogue_id_counter += 1
        self.dialogues[dialogue_id] = {
            "characters": [
                {"name": name1, "persona": persona1},
                {"name": name2, "persona": persona2},
            ],
            "location": {"name": location_name, "description": location_description},
            "history": [],
            "other": kwargs.get("other", {}),
        }
        return dialogue_id

    def end_dialogue(self, dialogue_id: str):
        """End a dialogue.

        Args:
            dialogue_id: ID of the dialogue.
        """
        del self.dialogues[dialogue_id]

    def generate_utterance(self, dialogue_id: str, speaker_id: str) -> str:
        """Generate an utterance for the given dialogue.

        Args:
            dialogue_id: ID of the dialogue.
            speaker: 0 for the first character, 1 for the second character.
        """
        context = self._build_context(dialogue_id, speaker_id)
        return self.text_generation_service.generate_utterance(context)

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
        scores = self.similarity_api.check_similarity(
            utterance, scripted_utterances, threshold
        )
        scores = [score if score > threshold else 0 for score in scores]
        if max(scores) > threshold:
            return scores.index(max(scores))
        return None

    def update_dialogue(self, dialogue_id: str, speaker_id: str, utterance: str):
        """Update dialogue state.

        Args:
            dialogue_id: ID of the dialogue.
            speaker_id: 0 for the first character, 1 for the second character.
            utterance: Natural language utterance.
        """
        self.dialogues[dialogue_id]["history"].append(
            {"speaker": speaker_id, "line": utterance}
        )

    def get_history(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """Get the history of a dialogue.

        Args:
            dialogue_id: ID of the dialogue.
        """
        return self.dialogues[dialogue_id]["history"]

    def _build_context(self, dialogue_id: str, speaker_id: str) -> Dict[str, Any]:
        """Build the context for the given dialogue.

        Args:
            dialogue_id: ID of the dialogue.
        """
        speaker = [c["name"] for c in self.dialogues[dialogue_id]["characters"]].index(
            speaker_id
        )
        other_speaker = speaker ^ 1
        context = copy(self.context_template)
        context["history"] = self.dialogues[dialogue_id]["history"]
        context["location"] = self.dialogues[dialogue_id]["location"]["description"]
        context["location_name"] = self.dialogues[dialogue_id]["location"]["name"]
        context["name"] = self.dialogues[dialogue_id]["characters"][speaker]["name"]
        context["persona"] = self.dialogues[dialogue_id]["characters"][speaker][
            "persona"
        ]
        context["other_name"] = self.dialogues[dialogue_id]["characters"][
            other_speaker
        ]["name"]
        context["other_persona"] = self.dialogues[dialogue_id]["characters"][
            other_speaker
        ]["persona"]
        return context

    def _validate_context_template(self, context_template: Dict[str, Any]):
        """Validate the context template.

        Args:
            context_template: Context template.
        """
        if "persona" not in context_template:
            raise ValueError("Context template must contain a persona.")
        if "name" not in context_template:
            raise ValueError("Context template must contain a name.")
        if "location" not in context_template:
            raise ValueError("Context template must contain a location.")
        if "location_name" not in context_template:
            raise ValueError("Context template must contain a location name.")
        if "other_name" not in context_template:
            raise ValueError("Context template must contain a other_name field.")
        if "other_persona" not in context_template:
            raise ValueError("Context template must contain a other_persona field.")
        if "history" not in context_template:
            raise ValueError("Context template must contain a history field.")
