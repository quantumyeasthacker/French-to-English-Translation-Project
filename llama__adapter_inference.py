"""
paired down version of llama_adapter.py which is for inferencing and calculating bleu score
"""

from typing import List, Optional
import fire
import fairscale
import sentencepiece
import pandas as pd
import numpy as np
import torch
import csv
import random

from gen_logits__ftw2 import *
from inference import text_completion
from datasets import load_metric
from torch.utils.data import DataLoader

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from adapter import Adapter


class config():
    def __init__(self, model_folder, model_name):
        self.model_folder = model_folder
        self.model_name = model_name

def get_weights_file_path(config_inst, epoch: str):
    model_fold = f"{config_inst.model_folder}"
    model_filename = f"{config_inst.model_name}{epoch}.pt"
    return str(Path('.') / model_fold / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config_inst):
    model_fold = f"{config_inst.model_folder}"
    model_filename = f"{config_inst.model_name}*"
    weights_files = list(Path(model_fold).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def save_loss_to_csv(loss_value, csv_filename):
    # Check if the file already exists
    try:
        with open(csv_filename, 'r') as file:
            # File exists, append to existing file
            reader = csv.reader(file)
            rows = list(reader)
            iteration_number = len(rows) + 1
    except FileNotFoundError:
        # File does not exist, create a new file with header
        iteration_number = 1
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['score'])

    # Append the new loss value
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([iteration_number, loss_value])



def main(
    ckpt_dir: str = 'llama-2-7b/',
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 120,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = 120,
):

    # initilize model
    BATCH_SIZE = max_batch_size
    ckpt_dir=ckpt_dir
    tokenizer_path=tokenizer_path
    max_seq_len=max_seq_len
    max_batch_size=max_batch_size
    seed=1
    model_parallel_size=None

    sacreblue = load_metric("sacrebleu")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")



    # freeze initial model (output layer removed)
    for p in model.parameters():
        p.requires_grad = False

    print(f'Initial parameter number: {sum(p.nelement() for p in model.parameters())}')


    # adding adapter module
    adapter = Adapter(model) # see adapter.py

    print(f'Number of parameters added: {sum(p.nelement() for p in adapter.parameters())}')

    # optimizer = torch.optim.SGD(adapter.parameters(), lr=0.005)

    # If model is specified for preload before training, load it
    model_folder = '/content/drive/MyDrive/10701project/llama/llama/saved_models'
    model_name = 'adapter'
    config_inst = config(model_folder, model_name)

    initial_epoch = 0
    preload = 'latest'
    model_filename = latest_weights_file_path(config_inst) if preload == 'latest' else get_weights_file_path(config_inst, preload) if preload else None
    if model_filename:
        print(f'Preloading adapter {model_filename}')
        state = torch.load(model_filename, map_location="cpu")
        adapter.load_state_dict(state['adapter_state_dict'])
        # initial_epoch = state['epoch'] + 1
        # optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        print('No adapter to preload, starting from pre-trained weights')
        init_params = {'adapter.weight': checkpoint['output.weight']} # initializing parameters with pretrained weights
        adapter.load_state_dict(init_params, strict=True)

 

    # data loading
    file_path = '/content/drive/MyDrive/10701project/en-fr.csv'

    # calculate bleu score and show text output for several sentences
    print('now inference!')
    data_test = pd.read_csv(file_path, nrows = int(200), skiprows=range(192_500), names=['en','fr'])
    pairs_test = list(zip(data_test['en'], data_test['fr']))
    test_dataloader = DataLoader(pairs_test, batch_size=BATCH_SIZE)

    bleu_scores = []
    for src, tgt in test_dataloader:
        targets: List[str] = [tgt]
        print('targets:', targets)

        prompts: List[str] = [
            'Translate English to French: sea otter => loutre de mer, peppermint => menthe poivrée, plush giraffe => girafe peluche, ' + str(x) + ' => ' for x in [src]
            ]
        print('prompts:', prompts)

        results = text_completion(
            model,
            tokenizer,
            adapter,
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            echo=False,
        )

        print('results')
        for t, r in zip(targets,results):
            print(r['generation'])
            score = sacreblue.compute(predictions=[r['generation']],
                                            references=[[t]])
            save_loss_to_csv(score['score'], 'bleu_scores.csv')
        print('bleu score:', score['score'])

    # plot histogram
    import pandas as pd
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.hist(bleu['score'])
    plt.ylabel('number of sentences')
    plt.xlabel('BLEU score')
    plt.show()
    fig.savefig('adapter_bleu_score_hist.pdf')


if __name__ == "__main__":
    fire.Fire(main)
