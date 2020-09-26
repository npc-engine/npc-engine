import numpy as np
import onnxruntime as rt
from tokenizers import ByteLevelBPETokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def test_model_compliance():
    rt_model = rt.InferenceSession(
        "chatbot_server/resources/models/gpt/gpt2.onnx",
        providers=[rt.get_available_providers()[1]],
    )

    pt_model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium').eval()

    # tok = ByteLevelBPETokenizer("chatbot_server/resources/tokenizer_data/vocab.json", "chatbot_server/resources/tokenizer_data/merges.txt")
    tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", fast=True)

    history = ["Hi, how are you doing?"]
    s = time.time()
    rt_generated = rt_argmax_gen(rt_model, tok, history=history)
    print("Elapsed time", time.time() - s)
    pt_generated = pt_argmax_gen(pt_model, tok, history=history)

    print("onnxrt: " + rt_generated)
    print("transformers: " + pt_generated)
    assert rt_generated == pt_generated


def rt_argmax_gen(model, tok, history, max_steps=25, min_length=2):
    utterance = []
    token = None
    for i in range(max_steps):
        total = history
        total = "<|endoftext|>".join(total) + "<|endoftext|>"
        ids = tok.encode(total)
        ids += utterance
        ids = np.asarray(ids).astype(np.int64).reshape([1, -1])
        o = model.run(None, {'input_ids': ids})
        logits = o[0][:, -1, :]
        if i < min_length:
            logits[:, -1] = -float("inf")
        token = np.argmax(logits, axis=-1)
        if token == tok.eos_token_id:
            break
        utterance += [token]
    return tok.decode(utterance)


def pt_argmax_gen(model, tok, history, max_steps=25, min_length=2):
    with torch.no_grad():
        utterance = []
        token = None
        for i in range(max_steps):
            if token == tok.eos_token_id:
                break
            total = history
            total = "<|endoftext|>".join(total) + "<|endoftext|>"
            # print(tok.encode(total))
            ids = tok.encode(total)
            ids += utterance
            ids = torch.LongTensor(ids).view([1, -1])
            o = model(**{'input_ids': ids})
            logits = o[0][:, -1, :]
            if i < min_length:
                logits[:, -1] = -float("inf")
            token = torch.argmax(logits, axis=-1).item()

            utterance += [token]
        return tok.decode(utterance)
