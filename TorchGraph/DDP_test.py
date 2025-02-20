"""
运行方式：
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 DDP_test.py
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 DDP_test.py

PS:不要忘记指定node_rank

"""

import json
import torch
import torchvision
from DDP_graph import DDPGraph
import torch.optim as optim
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--path', type=str, default='trace_file/origin-test-DDP.json',
                    help='path')
args = parser.parse_args()

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

from torchvision import models
module = getattr(models, args.model)().cuda()
example = torch.rand(32, 3, 224, 224).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)

g = DDPGraph(module, example, optimizer, 'resnet50', local_rank)
# for node in g.get_output_json():
#     print(node)
# g.dump_graph(args.model+'DDP.json')
g.dump_graph(args.path)


# from transformers import GPT2Model, GPT2Config
# config = GPT2Config(
#         n_embd=1280,
#         n_layer=24,
#         n_head=20,
#     )

# module = GPT2Model(config).cuda()
# example = (torch.LongTensor(1,224).random_() % 1000).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
