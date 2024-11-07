import sys
import math
import numpy as np
import torch
import pandas as pd
import random
import re
from pathlib import Path

from torch.nn import Transformer

from matplotlib import pyplot as plt
from torch import nn
import csv

from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.nn.utils import parametrize

import os
import tqdm

from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator

from gradient_descent_the_ultimate_optimizer import gdtuo
from model_Transformer import * # import transformer model

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'

file_path = '/content/drive/MyDrive/10701project/en-fr.csv'

data = pd.read_csv(file_path, nrows = int(75_000))

# function for saving files
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
            writer.writerow(['Iteration', 'Loss'])

"""
functions for tokenizing inputs
"""
SRC_LANGUAGE = 'fr'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = (str(s).lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def tokenizer(string) -> List[int]:
    string = normalizeString(string)
    return str(string).strip().split()

token_transform[SRC_LANGUAGE] = tokenizer
token_transform[TGT_LANGUAGE] = tokenizer

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    pairs = zip(data['fr'], data['en'])
    train_iter = iter(pairs)
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

len(vocab_transform['fr'])

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        if type(src_sample) or type(tgt_sample) is not str:
            src_sample = str(src_sample)
            tgt_sample = str(tgt_sample)
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# mask function for self attention
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# functions for finding latest saved weights
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

model_folder = '/content/drive/MyDrive/10701project/model'
model_name = 'nn_transformer_hyperoptTest'
config_inst = config(model_folder, model_name)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 256
BATCH_SIZE = 15
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


# If model is specified for preload before training, load it
initial_epoch = 0
preload = 'latest'
model_filename = latest_weights_file_path(config_inst) if preload == 'latest' else get_weights_file_path(config_inst, preload) if preload else None

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

from torch.utils.data import DataLoader

def train_epoch(mw, pairs, epoch, BATCH_SIZE):

    mw.train()
    losses = 0
    print('train!')
    train_dataloader = DataLoader(pairs, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        mw.begin()

        src = src.type(torch.LongTensor).to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = mw.forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        mw.zero_grad()
        loss.backward(create_graph=True)
        mw.step()

        losses += loss.item()
        print('loss:', loss.item())
        save_loss_to_csv(loss.item(), 'train_losses.csv')


    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config_inst, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'lr_state_dict': optim.parameters,
    }, model_filename)

    return losses / len(list(train_dataloader)), optim.parameters['alpha'].item()


def evaluate(model, pairs, BATCH_SIZE):
    model.eval()
    losses = 0

    print('eval!')
    val_dataloader = DataLoader(pairs, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        print('loss', loss.item())
        save_loss_to_csv(loss.item(), 'eval_losses.csv')


    return losses / len(list(val_dataloader))

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<sos>", "").replace("<eos>", "")

# data loading
data_train = pd.read_csv(file_path, nrows = int(10_000), skiprows=range(0), names=['en','fr'])
pairs_train = list(zip(data_train['fr'], data_train['en']))

data_val = pd.read_csv(file_path, nrows = int(1_000), skiprows=range(10_000), names=['en','fr'])
pairs_val = list(zip(data_val['fr'], data_val['en']))


# train model
from timeit import default_timer as timer
NUM_EPOCHS = 15


alpha_start = [6e-3,1e-3,1e-4,1e-5,1e-4]
alpha = np.zeros((len(alpha_start),NUM_EPOCHS))
train_loss = np.zeros((len(alpha_start),NUM_EPOCHS))
val_loss = np.zeros((len(alpha_start),NUM_EPOCHS))

for i in range(len(alpha_start)):
    print('initial alpha:', alpha_start[i])
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    optim = gdtuo.SGD(optimizer=gdtuo.SGD(alpha_start[i]))
    if i == 4:
        optim = gdtuo.SGD(alpha_start[i])
    mw = gdtuo.ModuleWrapper(model, optimizer=optim)
    mw.initialize()
    model.to(DEVICE)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss[i,epoch-1], alpha[i,epoch-1] = train_epoch(mw, pairs_train, epoch, BATCH_SIZE)
        val_loss[i,epoch-1] = evaluate(model, pairs_val, BATCH_SIZE)
        end_time = timer()
        print((f"Epoch: {epoch}, Train loss: {train_loss[i,epoch-1]:.3f}, Val loss: {val_loss[i,epoch-1]:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        print(translate(model, "Bonjour le monde. Je suis fatigué mais vivant."))

fig = plt.figure()
a_init = np.array([alpha_start])
al = np.concatenate((a_init, alpha.T))
al[1,0] = 100
plt.plot(range(NUM_EPOCHS+1), al, label=alpha_start)
plt.xlabel('EPOCH')
plt.ylabel('learning rate')
plt.legend(title='initial learning rate')
plt.yscale('log')
plt.show()
fig.savefig('learningRate.pdf')

fig = plt.figure()
plt.plot(range(1,NUM_EPOCHS+1), train_loss.T, label=alpha_start)
plt.xlabel('EPOCH')
plt.ylabel('train loss')
plt.yscale('log')
plt.legend(title='initial learning rate')
plt.show()
fig.savefig('trainloss.pdf')

fig = plt.figure()
plt.plot(range(1,NUM_EPOCHS+1), val_loss.T, label=alpha_start)
plt.xlabel('EPOCH')
plt.ylabel('test loss')
plt.yscale('log')
plt.legend(title='initial learning rate')
plt.show()
fig.savefig('testloss.pdf')

