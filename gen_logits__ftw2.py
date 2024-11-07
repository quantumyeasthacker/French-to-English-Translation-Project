from typing import List, Optional
import torch
"""
this function tokenizes the input, feeds it through the frozen pre-trained model,
and then passes it through output layer (which is trained) to generate the logits
"""

def generate_logits(
    model,
    tokenizer,
    adapter,
    prompt_tokens: List[List[int]],
    max_gen_len: int,
):

    params = model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    # truncate sentence if too long
    prompt_tokens = [t[:params.max_seq_len] if len(t) >= params.max_seq_len else t for t in prompt_tokens]

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.eos_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0
    if min_prompt_len == total_len:
        print('warning: min_prompt_len == total_len')

    # forward pass through pre-trained model
    model_output = model.forward(tokens, prev_pos)
    # convert from inference tensor to normal tensor which can store gradients
    output = model_output.clone().detach().requires_grad_(True)
    # forward pass through adapter
    logits = adapter.forward(output)

    return logits.transpose(1,2)