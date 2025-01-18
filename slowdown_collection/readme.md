# Slowdown Cases Collection Module
This module collects slowdown cases using NVIDIA Nsight Systems (nsys).

## Usage
1. Prepare `train_script.py`, `define_model.py`, `local_config.json`, `global_config.json` in `input` folder.

2. Execute shell script:
```
run.sh
```

## Example Configurations
All the configurations should be placed into the `input` folder.

### Example `global_config.json`
```
{
	"cuda_visible_devices": "0,1",
	"cuda_version_check": "11.8",
	"nsys_path": "/home/eric/project/cuda-toolkit-12_6/cuda-toolkit-12.6/nsight-systems-2024.5.1/bin/nsys",
	"python_path": "/home/eric/miniconda3/envs/torchgraph/bin/python"
}
```

|Key|Value|
|-|-|
|cuda_visible_devices| Set the visible devices parameter|
|cuda_version_check|cuda version of PyTorch|
|nsys_path|Path to NVIDIA Nsight Systems (nsys) executable, can check with "which nsys"|
|python_path|Path to python executable, can check with "which python"|

### Example `local_config.json`
```
{
  "cap_size": 2,
  "epochs": 3,
  "batchsize": 16
}
```

|Key|Value|
|-|-|
|cap_size|bucket_cap_mb setting in PyTorch DDP|
|epochs|Total number of epochs, we only profile results from the middle od the last epoch|
|batchsize|Number of training samples in a single pass|

### Example `define_model.py`
```
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
```

|Function|Description|Input Value|Expected Return Value|
|-|-|-|-|
|get_model(rank, cap_size)|Return a model callable for training|rank, cap_size|a model callable for training|
|get_dataloader(rank, batchsize)|Return a simple dataloader for training|rank, batchsize|data_loader|a data loader for training|
|forward_pass(model, inputs)|Run a forward pass|model, inputs|output of the forward pass|
|backward_pass(output)|Run a backward pass|output|N/A|