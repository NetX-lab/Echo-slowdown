from typing import Any

import numpy as np
import random
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

__all__ = [
    "gpt2_medium",
    "gpt2_large",
    "gpt2_xl",
]

class GPT2BenchmarkModel(torch.nn.Module):
    """The GPT2 model for benchmarking."""
    def __init__(self, config, num_classes=1000):
        """Constructor.
        Args:
            config (GPT2Config): Configurations of GPT2 model.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._model = GPT2Model(config)
        self._linear = torch.nn.Linear(config.n_embd, num_classes)

    def forward(self, input):
        """Forward propagation function.
        Args:
            input (torch.LongTensor): Indices of input sequence tokens in the vocabulary,
              shape (batch_size, sequence_length).
        Return:
            result (torch.FloatTensor): Last layer hidden-state of the first token of the sequence
              (classification token) further processed by a Linear layer, shape (batch_size, hidden_size).
        """
        outputs = self._model(input)
        result = self._linear(outputs[0])
        return result

def gpt2_medium(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12
    )
    return GPT2BenchmarkModel(config, **kwargs)

def gpt2_large(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=1024, n_layer=24, n_head=16
    )
    return GPT2BenchmarkModel(config, **kwargs)

def gpt2_xl(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=1600, n_layer=48, n_head=25
    )
    return GPT2BenchmarkModel(config, **kwargs)

