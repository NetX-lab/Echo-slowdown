from typing import Any

import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

__all__ = [
    "bert_base",
    "bert_large",
]

class BertBenchmarkModel(torch.nn.Module):
    """The GPT2 model for benchmarking."""
    def __init__(self, config, num_classes=1000):
        """Constructor.
        Args:
            config (GPT2Config): Configurations of GPT2 model.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._model = BertModel(config)
        self._linear = torch.nn.Linear(config.hidden_size, num_classes)

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
        result = self._linear(outputs[1])
        return result

def bert_base(sample = False,  **kwargs: Any):
    config = BertConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
    )
    return BertModel(config)

def bert_large(sample = False,  **kwargs: Any):
    config = BertConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096
    )
    return BertModel(config)
