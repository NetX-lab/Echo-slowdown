from typing import Any

import numpy as np
import random
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

__all__ = [
    "t5",
]

class T5BenchmarkModel(torch.nn.Module):
    """The T5 model for benchmarking."""
    def __init__(self, config, num_classes=1000):
        """Constructor.
        Args:
            config (T5Config): Configurations of T5 model.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._model = T5ForConditionalGeneration(config)
        self._linear = torch.nn.Linear(config.d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        """Forward propagation function.
        Args:
            input_ids (torch.LongTensor): Indices of input sequence tokens in the vocabulary,
              shape (batch_size, sequence_length).
            attention_mask (torch.LongTensor, optional): Mask to avoid performing attention on padding tokens.
            decoder_input_ids (torch.LongTensor): Indices of target sequence tokens in the vocabulary,
              shape (batch_size, target_sequence_length).
        Return:
            result (torch.FloatTensor): Last layer hidden-state of the first token of the sequence
              (classification token) further processed by a Linear layer, shape (batch_size, num_classes).
        """
        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        result = self._linear(outputs.logits[:, 0, :])  # Using logits from the first token
        return result

def t5(sample=False, **kwargs: Any):
    config = T5Config(
        d_model=768, num_layers=12, num_heads=12, d_ff=3072, 
        dropout_rate=0.1, activation_function="relu"
    )
    return T5ForConditionalGeneration(config)
