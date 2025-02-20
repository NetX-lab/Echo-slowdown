
'''
./native_torch_run.sh

export CUDA_VISIBLE_DEVICES=1,2,3,6

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ../trace_file/ddp_tes_ori \
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 ddp_run.py
'''
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.nvtx as nvtx
import os
import argparse
from torch.profiler import profile, schedule, ProfilerActivity, record_function

from example_model import SimpleModel


def cleanup():
    dist.destroy_process_group()



def train_ddp(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")

    # model = SimpleModel().cuda()
    # example_input = torch.randn(1, 512).cuda()
    from torchvision import models
    model = getattr(models, args.model)().cuda()
    example_input = torch.rand(32, 3, 224, 224).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    ddp_model = DDP(model, bucket_cap_mb=1)
    print("ddp_model = DDP(model)") if rank == 0 else 0

    ddp_model.train()
    for epoch in range(5):
        nvtx.range_push("iteration:"+str(epoch))

        nvtx.range_push("forward prop")
        outputs = ddp_model(example_input)
        loss = outputs.sum()
        nvtx.range_pop()

        nvtx.range_push("backward prop")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("param optim")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_pop()

        print(f"Epoch: {epoch}, Loss: {loss.item()}") if rank == 0 else 0

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDP test')
    parser.add_argument('--batch_size', default=-1, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    train_ddp(local_rank, torch.cuda.device_count())

