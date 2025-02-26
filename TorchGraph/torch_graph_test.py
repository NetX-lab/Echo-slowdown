# import json
# import torch
# import torchvision
# from torch_graph import TorchGraph
# import torch.optim as optim
# import argparse


# parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--model', type=str, default='resnet50',
#                     help='model to benchmark')
# args = parser.parse_args()

# args.model="gpt2"
# args.type="NLP"

# from torchvision import models
# module = getattr(models, args.model)().cuda()
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
# g.dump_graph(args.model + "test.json")

import json
import torch
import torchvision
from torch_graph import TorchGraph
import torch.optim as optim
import argparse

'''
没被切割(TP、PP)的模型的完整过程可以通过class torch_graph构建

Q:
    1. 异步情况是否包含?
    2. NCCL通信是否已经记录在grad中?
    3. overlap情况如果正常发生,单从新加入属性1无法确认cpu和kernel的过程,但是如果可以确认kernel发生在这个时间段,具体的时间点或许不重要?考虑到可能并发执行的只有
    不同model中,如果记录的operation除了id完全一致,是否其执行时间也一致呢？
    4. 对于一个新设计的model,怎么样的模拟DAG才能还原最可能的overlap发生情况?因为预测overlap的影响本身是建立在overlap发生的基础上的


需要添加的属性:
    1. start/end point (time)
    2. which layer (layer1 -> [..operations...] -> layer2), so operations belong to layer1
    3. which rank
'''

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--type', type=str, default='CV',
                    help='model types')
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--path', type=str, default='mytest_graph_simplemodel.json',
                    help='path')
parser.add_argument('--path_var', type=str, default='DDP.json',
                    help='path')
args = parser.parse_args()

from torchvision import models
import transformer
import time

args.model="resnet50"
args.type="CV"
# args.path_var="0515_simple_module.json"

model = args.model
# timer = Timer(100, args.model)
# if args.type == 'CV':
    # module = getattr(models, args.model)().cuda()
    # module = getattr(models, args.model)().cpu()
    # example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    # example = torch.rand(args.batchsize, 3, 224, 224).cpu()
    # optimizer = optim.SGD(module.parameters(), lr=0.01)

# elif args.type == 'NLP':
    # module = getattr(transformer, args.model)().cuda()
    # module = getattr(transformer, args.model)().cpu()
    # example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
    # example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cpu()
    # optimizer = optim.SGD(module.parameters(), lr=0.01)


import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(10*32*32, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

module = SimpleModel()
example = torch.randn(1, 3, 32, 32) # .cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)


""" 测试DDP类型的完整fwd+bwd graph """
time_start = time.time()
g = TorchGraph(module=module, example=example, optimizer=None, name='SimpleModel')
time_cume = time.time() - time_start
print(f"cost time = {time.time() - time_start}")

g.dump_graph(args.path)
print("torch_graph: 已完成dump_graph...")


""" 测试单个fwd/bwd graph """
# time_start = time.time()
# g = TorchGraph(module, example, optimizer, 'SimpleModel')
# time_cume = time.time() - time_start
# print(f"cost time = {time.time() - time_start}")

g.dump_fwd_graph('mytest_fwd_graph.json')
print("torch_graph: 已完成mytest_fwd_graph...")

g.dump_bwd_graph('mytest_bwd_graph.json')
print("torch_graph: 已完成mytest_bwd_graph...")

g.dump_graph('mytest_fbwd_graph.json')
print("torch_graph: 已完成mytest_fbwd_graph...")