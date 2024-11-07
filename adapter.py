"""
output module which is being trained
"""

import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import ColumnParallelLinear

class Adapter(torch.nn.Module):
    def __init__(self, model,):
        super().__init__()
        self.params = model.params
        self.vocab_size = model.params.vocab_size
        self.adapter = ColumnParallelLinear(
            model.params.dim, model.params.vocab_size, bias=False
        )
    
    def forward(self, output):
        adapter = self.adapter(output).float()
        return adapter