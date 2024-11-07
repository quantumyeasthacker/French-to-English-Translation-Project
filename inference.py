from typing import List, Literal, Optional, Tuple, TypedDict
from google.colab import output
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
For testing quality of translation, uses sqential decoding for model inference
"""

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    adapter,
    prompt_tokens: List[List[int]],
    max_gen_len: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

    params = model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    # truncate sentence if too long
    prompt_tokens = [t[:params.max_seq_len] if len(t) >= params.max_seq_len else t for t in prompt_tokens]

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = model.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens,
            reduction="none",
            ignore_index=pad_id,
        )

    for cur_pos in range(min_prompt_len, total_len):
        output = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        logits = adapter.forward(output)

        # if temperature > 0:
        #     probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        #     next_token = sample_top_p(probs, top_p)
        # else:
        next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        if logprobs:
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1],
                reduction="none",
                ignore_index=pad_id,
            )
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eos_id
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if tokenizer.eos_id in toks:
            eos_idx = toks.index(tokenizer.eos_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)




def text_completion(
    model,
    tokenizer,
    adapter,
    prompts: List[str],
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    logprobs: bool = False,
    echo: bool = False,
) -> List[CompletionPrediction]:
    if max_gen_len is None:
            max_gen_len = model.params.max_seq_len - 1
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens, generation_logprobs = generate(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        echo=echo,
    )
    if logprobs:
        return [
            {
                "generation": tokenizer.decode(t),
                "tokens": [tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
    return [{"generation": tokenizer.decode(t)} for t in generation_tokens]
