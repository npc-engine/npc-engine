"""Module that implements Huggingface transformers semantic similarity."""
from typing import List
import numpy as np
from os import path
from tokenizers import Tokenizer
import onnxruntime as rt
from onnxruntime import GraphOptimizationLevel as opt_level
from loguru import logger
from pyctcdecode import build_ctcdecoder
import librosa

from npc_engine.services.stt.stt_base import SpeechToTextAPI


class NemoSTT(SpeechToTextAPI):
    """Text to speech pipeline based on Nemo toolkit.

    Uses:

        - ONNX export of EncDecCTCModel from Nemo toolkit.
        - Punctuation distillbert model from Nemo toolkit. (requires tokenizer.json as well)
        - Huggingface transformers model for predicting that sentence is finished
            (Cropped sentence -> 0 label, finished sentence -> 1 label).
        - OpenSLR Librispeech 3-gram model converted to lowercase https://www.openslr.org/11/

    References:
        https://github.com/NVIDIA/NeMo
        https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5
        https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html


    ctc.onnx spec:

        - inputs:
            `audio_signal` mel spectogram of shape `(batch_size, 64, mel_sequence)`
        - outputs:
            `tokens` of shape `(batch_size, token_sequence, logits)`

    punctuation.onnx spec:

        - inputs:
            `input_ids` mel spectogram of shape `(batch_size, sequence)`
            `attention_mask` mel spectogram of shape `(batch_size, sequence)`
        - outputs:
            `punctuation` of shape `(batch_size, sequence, 4)`
            `capitalization` of shape `(batch_size, sequence, 2)`
    """

    def __init__(
        self,
        model_path: str,
        frame_size: int = 1000,
        sample_rate: int = 16000,
        predict_punctuation: bool = False,
        *args,
        **kwargs,
    ):
        """Create and load biencoder model for semantic similarity.

        Args:
            model_path: A path where model config and weights are stored.
            frame_size: Size of the audio frame in milliseconds.
            sample_rate: Sample rate of the audio.
            predict_punctuation: Whether to predict punctuation and capitalization.
        """
        super().__init__(
            sample_rate=sample_rate,
            model_path=model_path,
            frame_size=frame_size,
            *args,
            **kwargs,
        )
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_BASIC
        provider = rt.get_available_providers()[0]
        logger.info(f"STT uses {provider} provider")

        self.stft_filterbanks = librosa.filters.mel(
            16000, 512, n_mels=64, fmin=0, fmax=8000
        )
        self.stft_window = librosa.filters.get_window("hann", 320, fftbins=False)
        self.mel_mean, self.mel_std = self._fixed_normalization()

        self.asr_model = rt.InferenceSession(
            path.join(model_path, "ctc.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )

        self.predict_punctuation = predict_punctuation
        if self.predict_punctuation:
            self.punctuation = rt.InferenceSession(
                path.join(model_path, "punctuation.onnx"),
                providers=self.get_providers(),
                sess_options=sess_options,
            )
            self.tokenizer = Tokenizer.from_file(
                path.join(model_path, "tokenizer.json")
            )
        self.asr_vocab = " abcdefghijklmnopqrstuvwxyz'"
        self.asr_vocab = list(self.asr_vocab)
        self.asr_vocab.append("")

        self.punct_labels = "O,.?"
        self.capit_labels = "OU"

        self.decoder = build_ctcdecoder(self.asr_vocab)

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = opt_level.ORT_ENABLE_ALL
        self.sentence_model = rt.InferenceSession(
            path.join(model_path, "sentence_prediction.onnx"),
            providers=self.get_providers(),
            sess_options=sess_options,
        )
        self.sentence_tokenizer = Tokenizer.from_file(
            path.join(model_path, "sentence_tokenizer.json")
        )

        logger.info(
            f"Sentence classifier uses {rt.get_available_providers()[0]} provider"
        )

    def transcribe(self, audio: List[float]) -> np.ndarray:
        """Transcribe audio usign this pipeline.

        Args:
            audio: ndarray of int16 of shape (samples,).

        Returns:
            Transcribed text from the audio.
        """
        logits = self._predict(np.asarray(audio, dtype=np.float32))
        return logits

    def decode(self, logits: np.ndarray) -> str:
        """Decode logits into text.

        Args:
            logits: ndarray of float32 of shape (timesteps, vocab_size).

        Returns:
            Decoded string.
        """
        return self.decoder.decode(logits)

    def decide_finished(self, context: str, text: str) -> bool:
        """Decide if audio transcription should be finished.

        Args:
            context: Text context of the speech recognized
                (e.g. a question to which speech recognized is a reply to).
            text: Recognized speech so far

        Returns:
            Decision to stop recognition and finalize results.
        """
        decision = self._decide_sentence_finished(context, text)
        done = bool(decision)
        return done

    def postprocess(self, text: str) -> str:
        """Add punctuation and capitalization.

        Args:
            text: audio transcription.

        Returns:
            Postprocessed text transcribtion.
        """
        if self.predict_punctuation:
            enc = self.tokenizer.encode(text)

            punct, capit = self.punctuation.run(
                None,
                {
                    "input_ids": np.asarray(enc.ids, np.int64).reshape([1, -1]),
                    "attention_mask": np.asarray(enc.attention_mask, np.int64).reshape(
                        [1, -1]
                    ),
                },
            )
            punct = punct.argmax(2)
            capit = capit.argmax(2)
            punctuated_capitalized = self._apply_punct_capit_predictions(
                text, punct[0][1:-1].tolist(), capit[0][1:-1].tolist()
            )
            return punctuated_capitalized
        else:
            return text

    def _decide_sentence_finished(self, context, text):
        tokenized = self.sentence_tokenizer.encode(context, text)
        ids = np.asarray(tokenized.ids).reshape([1, -1]).astype(np.int64)
        type_ids = np.asarray(tokenized.type_ids).reshape([1, -1]).astype(np.int64)
        attention_mask = np.ones_like(ids)
        input_dict = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }
        logits = self.sentence_model.run(None, input_dict)[0]
        return logits.argmax(-1)[0]

    def _predict(self, audio: np.ndarray) -> np.ndarray:
        signal = audio.reshape([1, -1])
        audio_signal = self._preprocess_signal(signal).astype(np.float32)
        return self.asr_model.run(None, {"audio_signal": audio_signal})[0][0]

    def _preprocess_signal(self, signal):
        audio_signal = signal.reshape([1, -1])
        # audio_signal += np.random.rand(*audio_signal.shape) * 1e-5
        audio_signal = np.concatenate(
            (
                audio_signal[:, 0].reshape([-1, 1]),
                audio_signal[:, 1:] - 0.97 * audio_signal[:, :-1],
            ),
            axis=1,
        )
        audio_signal = audio_signal.reshape([-1])
        spectogram = librosa.stft(
            audio_signal,
            n_fft=512,
            hop_length=160,
            win_length=320,
            window=self.stft_window,
            center=True,
        )
        spectogram = np.stack([spectogram.real, spectogram.imag], -1)

        spectogram = np.sqrt((spectogram ** 2).sum(-1))
        spectogram = spectogram ** 2
        spectogram = np.dot(self.stft_filterbanks, spectogram)
        spectogram = np.expand_dims(spectogram, 0)
        spectogram = np.log(spectogram + (2 ** -24))

        spectogram = spectogram - np.asarray(self.mel_mean).reshape([1, 64, 1])
        spectogram = spectogram / np.asarray(self.mel_std).reshape([1, 64, 1])
        return spectogram

    def _apply_punct_capit_predictions(
        self, query: str, punct_preds, capit_preds
    ) -> str:
        """Restores punctuation and capitalization in ``query``.

        Args:
            query: a string without punctuation and capitalization
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            a query with restored punctuation and capitalization
        """
        query = query.strip().split()

        query_with_punct_and_capit = ""
        skip = 0
        for j, word in enumerate(query):
            word_enc = self.tokenizer.encode(word, add_special_tokens=False).ids
            skip += len(word_enc) - 1
            punct_label = self.punct_labels[punct_preds[j + skip]]
            capit_label = self.capit_labels[capit_preds[j + skip]]

            if capit_label != "O":
                word = word.capitalize()
            query_with_punct_and_capit += word
            if punct_label != "O":
                query_with_punct_and_capit += punct_label
            query_with_punct_and_capit += " "
        return query_with_punct_and_capit[:-1]

    def _fixed_normalization(self):
        """From https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_ASR_Microphone_Demo.ipynb .

        For whatever reason they use these values to normalize the mel spectogram.
        """
        mel_fixed_mean = [
            -14.95827016,
            -12.71798736,
            -11.76067913,
            -10.83311182,
            -10.6746914,
            -10.15163465,
            -10.05378331,
            -9.53918999,
            -9.41858904,
            -9.23382904,
            -9.46470918,
            -9.56037,
            -9.57434245,
            -9.47498732,
            -9.7635205,
            -10.08113074,
            -10.05454561,
            -9.81112681,
            -9.68673603,
            -9.83652977,
            -9.90046248,
            -9.85404766,
            -9.92560366,
            -9.95440354,
            -10.17162966,
            -9.90102482,
            -9.47471025,
            -9.54416855,
            -10.07109475,
            -9.98249912,
            -9.74359465,
            -9.55632283,
            -9.23399915,
            -9.36487649,
            -9.81791084,
            -9.56799225,
            -9.70630899,
            -9.85148006,
            -9.8594418,
            -10.01378735,
            -9.98505315,
            -9.62016094,
            -10.342285,
            -10.41070709,
            -10.10687659,
            -10.14536695,
            -10.30828702,
            -10.23542833,
            -10.88546868,
            -11.31723646,
            -11.46087382,
            -11.54877829,
            -11.62400934,
            -11.92190509,
            -12.14063815,
            -11.65130117,
            -11.58308531,
            -12.22214663,
            -12.42927197,
            -12.58039805,
            -13.10098969,
            -13.14345864,
            -13.31835645,
            -14.47345634,
        ]
        mel_fixed_std = [
            3.81402054,
            4.12647781,
            4.05007065,
            3.87790987,
            3.74721178,
            3.68377423,
            3.69344,
            3.54001005,
            3.59530412,
            3.63752368,
            3.62826417,
            3.56488469,
            3.53740577,
            3.68313898,
            3.67138151,
            3.55707266,
            3.54919572,
            3.55721289,
            3.56723346,
            3.46029304,
            3.44119672,
            3.49030548,
            3.39328435,
            3.28244406,
            3.28001423,
            3.26744937,
            3.46692348,
            3.35378948,
            2.96330901,
            2.97663111,
            3.04575148,
            2.89717604,
            2.95659301,
            2.90181116,
            2.7111687,
            2.93041291,
            2.86647897,
            2.73473181,
            2.71495654,
            2.75543763,
            2.79174615,
            2.96076456,
            2.57376336,
            2.68789782,
            2.90930817,
            2.90412004,
            2.76187531,
            2.89905006,
            2.65896173,
            2.81032176,
            2.87769857,
            2.84665271,
            2.80863137,
            2.80707634,
            2.83752184,
            3.01914511,
            2.92046439,
            2.78461139,
            2.90034605,
            2.94599508,
            2.99099718,
            3.0167554,
            3.04649716,
            2.94116777,
        ]
        return mel_fixed_mean, mel_fixed_std
