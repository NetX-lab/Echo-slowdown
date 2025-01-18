import os 
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision import models
import transformer
import json  # For reading the config files
import time
import sys

sys.path.append(os.path.dirname(__file__))
import define_model

# Set up the process group
def setup(rank, world_size):
    print(f"debug: Rank {rank}: Setting up process group")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"debug: Rank {rank}: Process group setup complete")

# Clean up the process group
def cleanup():
    print("debug: Cleaning up process group")
    dist.destroy_process_group()

# Prepare model and random data
def prepare(rank, world_size, cap_size, batchsize):
    print(f"debug: Rank {rank}: Preparing model and data")
    setup(rank, world_size)

    model = define_model.get_model(rank, cap_size)
    data_loader = define_model.get_dataloader(rank, batchsize)

    return model, data_loader


# Training function
def train(model, data_loader, rank, epochs, profile_model_iteration_time=False, model_iteration_time_output="", warmup_epochs=0):
    print(f"debug: Rank {rank}: Starting training for {epochs} epochs")
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    nb_iters = len(data_loader) * epochs  # Total number of iterations for the specified epochs
    last_epoch_start = nb_iters - len(data_loader)  # Start iteration of the last epoch
    middle_of_last_epoch = last_epoch_start + (len(data_loader) // 2)  # Middle iteration of the last epoch

    model.train()
    for epoch in range(warmup_epochs+epochs):  # Use the epochs argument to control training duration
        if epoch < warmup_epochs:
            print(f"debug: Rank {rank}: Warmup epoch {epoch+1}/{warmup_epochs} starting")
        else:
            print(f"debug: Rank {rank}: Actual epoch {epoch+1-warmup_epochs}/{epochs} starting")
        
        if epoch == warmup_epochs:
            start_time = time.time()

        for i, (inputs, labels) in enumerate(data_loader):
            global_iter = i + epoch * len(data_loader)

            optimizer.zero_grad()
            
            # Forward pass
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("forward")
                
            output = define_model.forward_pass(model, inputs)

            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_pop()

            # Start profiling at the middle of the last epoch and only backward pass (overlap only happens in backward pass)
            if global_iter == middle_of_last_epoch:
                print(f"debug: Rank {rank}: Starting backward pass profiling")
                torch.cuda.cudart().cudaProfilerStart()
                
            # Backward pass
            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_push("backward")

            define_model.backward_pass(output)

            if global_iter == middle_of_last_epoch:
                torch.cuda.nvtx.range_pop()

            # Stop profiling
            if global_iter == middle_of_last_epoch:
                print(f"debug: Rank {rank}: Stopping profiler")
                torch.cuda.cudart().cudaProfilerStop()

            # Optimize & update gradient (don't profile)
            optimizer.step()
        
        # Optional: Clear cache to manage memory usage
        torch.cuda.empty_cache()
        print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Finished epoch.")

    # If profile mode is model_time, write model time to a file
    if profile_model_iteration_time:
        end_time = time.time()
        execution_time = end_time - start_time
        with open(model_iteration_time_output, "w") as file:
            print(f"Execution time for {epochs} epochs: {execution_time:.6f} seconds", file=file)
            avg_execution_time = execution_time / epochs
            print(f"Average execution time per single iteration: {avg_execution_time:.6f} seconds", file=file)

def run_training(rank, world_size, cap_size, epochs, batchsize, profile_model_iteration_time=False, model_iteration_time_output="", warmup_epochs=0):
    print(f"debug: Rank {rank}: Running training process")
    model, data_loader = prepare(rank, world_size, cap_size, batchsize)
    train(model, data_loader, rank, epochs, profile_model_iteration_time=profile_model_iteration_time, model_iteration_time_output=model_iteration_time_output, warmup_epochs=warmup_epochs)
    cleanup()
    print(f"debug: Rank {rank}: Training process completed")

# Merge configurations from two config files
def merge_configs(config1, config2):
    merged_config = {**config1, **config2}
    return merged_config

# Main function
def main():
    parser = argparse.ArgumentParser(description="Profile training with DDP")
    parser.add_argument("--world_size", type=int, required=True, help="A mandatory integer argument representing the world size.")
    parser.add_argument("--local_config_file", type=str, required=True, help="Path to the primary configuration file")
    parser.add_argument("--global_config_file", type=str, required=True, help="Path to the global configuration file")
    
    args = parser.parse_args()

    # Load the primary configuration file
    with open(args.local_config_file, 'r') as f:
        config = json.load(f)
    
    # Load the global configuration file
    with open(args.global_config_file, 'r') as f:
        global_config = json.load(f)

    # Merge the three configurations
    final_config = merge_configs(config, global_config)

    # CUDA Version check
    if final_config['cuda_version_check']:
        if torch.version.cuda != final_config['cuda_version_check']:
            print(f'PyTorch CUDA version mismatch with CUDA version required in environment config!')
            print(f'Expected {final_config['cuda_version_check']}, but using {torch.version.cuda}.')
            raise("Exception")
        else:
            print(f'PyTorch CUDA version: {torch.version.cuda} matches with CUDA version required in environment config.')

    # Extract parameters from the final configuration
    world_size = args.world_size
    cap_size = final_config['cap_size']
    epochs = final_config['epochs']
    batchsize = final_config['batchsize']

    # Print specific configuration values for debugging
    print("Configuration Values:")
    print(f"cuda version check: {final_config['cuda_version_check']}")
    print(f"world_size: {world_size}")
    print(f"cap_size: {cap_size}")
    print(f"epochs: {epochs}")
    print(f"batchsize: {batchsize}")

    # profile mode model_time
    profile_model_iteration_time = False
    model_iteration_time_output = ""
    warmup_epochs = 0
    
    if 'warmup_epochs' in final_config:
        warmup_epochs = final_config['warmup_epochs']
    
    print("debug: Starting mp.spawn...")
    mp.spawn(run_training, args=(world_size, cap_size, epochs, batchsize, profile_model_iteration_time, model_iteration_time_output, warmup_epochs), nprocs=world_size, join=True)
    print("debug: mp.spawn completed")
    
if __name__ == '__main__':
    main()