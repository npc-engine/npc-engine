from typing import Dict, List, Any
from .text_generation import GPTTextGenerator
from .speech_synthesis import TacotronSpeechSynthesizer


class Chatbot:

    def __init__(self, gpt_path, tacotron_path):
        self.text_generator = GPTTextGenerator(gpt_path)
        self.speech_synthesizer = TacotronSpeechSynthesizer(tacotron_path)
        self.INCORRECT_MSG = 1

    def handle_message(self, message):
        if message['cmd'] == "create_speaker":
            return self.create_speaker(message)
        elif message['cmd'] == "step_dialog":
            return self.step_dialog(message)
        elif message['cmd'] == "script_line":
            return self.script_line(message)
        elif message['cmd'] == "delete_speaker":
            return self.delete_speaker(message)

    def create_speaker(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ['speaker_id', 'persona']):
            return {'status': self.INCORRECT_MSG}

        self.speech_synthesizer.create_voice(
            message['speaker_id'], message['traits']
            if 'traits' in message.keys() else None
        )
        self.text_generator.add_speaker(
            message['speaker_id'], message['persona']
        )
        return {'status': 0}

    def step_dialog(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(message, ['speaker_id', 'line']):
            return {'status': self.INCORRECT_MSG}
        reply, script_triggered = self.text_generator.step_dialog(
            message['speaker_id'], message['line']
        )
        audio_clip = self.speech_synthesizer.tts(message['speaker_id'], reply)
        return {
            'reply': audio_clip.reshape([-1]).tolist(),
            'script_triggered': script_triggered,
            'status': 0
        }

    def script_line(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(
            message, ['speaker_id', 'cue_line', 'script_line']
        ):
            return {'status': self.INCORRECT_MSG}
        self.text_generator.script_response(
            message['speaker_id'], message['cue_line'], message['script_line']
        )
        return {'status': 0}

    def delete_speaker(self, message: Dict[str, Any]):
        if not self._validate_msg_fields(
            message, ['speaker_id']
        ):
            return {'status': self.INCORRECT_MSG}
        self.text_generator.delete_speaker(message['speaker_id'])
        return {'status': 0}

    def _validate_msg_fields(self, msg: Dict[str, Any], fields: List[str]) -> bool:
        msg_correct = True
        for field in fields:
            msg_correct = msg_correct and (field in msg)
        return msg_correct
