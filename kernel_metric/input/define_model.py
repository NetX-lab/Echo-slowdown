import transformer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def get_model(rank, cap_size):
    model_name = 'gpt2'
    model = getattr(transformer, model_name)().to(rank)
    
    # Wrap model with DDP, using the bucket_cap_mb from the config
    model = DDP(model, device_ids=[rank], bucket_cap_mb=cap_size)

    return model

def get_dataloader(rank, batchsize):
    example = (torch.LongTensor(batchsize, 512).random_() % 1000).to(rank)
    
    # Create fake labels (assuming 10 classes for CV and NLP tasks)
    labels = torch.randint(0, 10, (batchsize,)).to(rank)
    
    # Use a simple data loader-like structure
    data_loader = [(example, labels)] * 100  # Simulate 100 batches
    
    return data_loader

def forward_pass(model, inputs):
    output = model(inputs)
    return output

def backward_pass(output):
    output.last_hidden_state.backward(output.last_hidden_state)