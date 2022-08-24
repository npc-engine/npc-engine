"""Utility functions for the text generation service."""
from typing import List, Tuple
import numpy as np
import scipy.special as scp


def decode_logits(
    logits: np.ndarray, temperature: float, topk: int, log_probs: List[float]
) -> Tuple[int, List[float]]:
    """Decode logits to token.

    Args:
        logits: Logits to decode of shape (batch, vocab_size,)
        temperature: Temperature parameter for sampling.
        topk: If not none selects top n of predictions to sample from during generation.
        log_probs: Cummulative logarithm of sequence probabilities for previous tokens in the sequence.

    Returns:
        Tuple of (token, log_probs)
        where token is the decoded token and sequence_probabilities are update
    """
    tokens = []
    for i, logit in enumerate(logits):
        if topk is not None:
            ind = np.argpartition(logit, -topk)[-topk:]
            new_logits = np.zeros(logit.shape)
            new_logits[ind] = logit[ind]
            logit = new_logits

        probs = scp.softmax(logit / temperature, axis=0)
        token = np.random.choice(np.arange(probs.shape[0]), p=probs)
        if np.isnan(np.log2(probs[token])):
            log_probs[i] += -10
        else:
            log_probs[i] += np.log2(probs[token])
        token = token.ravel()[0]
        tokens.append(token)
    return tokens, log_probs
