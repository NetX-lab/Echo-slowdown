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
    "gpt2"
]

def gpt2(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12
    )
    return GPT2Model(config)

def gpt2_medium(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12
    )
    return GPT2Model(config)

def gpt2_large(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=1024, n_layer=24, n_head=16
    )
    return GPT2Model(config)

def gpt2_xl(sample = False,  **kwargs: Any):
    config = GPT2Config(
        n_embd=1600, n_layer=48, n_head=25
    )
    return GPT2Model(config)

