import json
import torch
import torchvision
from torch_database import TorchDatabase
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time

import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--path', type=str, default='mytest_database_simplemodel.json',
                    help='path')
args = parser.parse_args()

from torchvision import models
# module = getattr(models, args.model)().cuda()
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# module = torchvision.models.resnet101(pretrained=True).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)
# example = torch.rand(32, 3, 224, 224).cuda()


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

timer = Timer(100, '0_model')
# time_start = time.time()
# g = TorchDatabase(module, example, 'SimpleModel', timer, optimizer)
# time_cume = time.time() - time_start
# print(f"cost time = {time.time() - time_start}")

# g.dump_graph(args.path)
# print("torch_database: 已完成dump_graph...")


g = TorchDatabase(module=module, example=example, name=f"0_model", timer=timer, optimizer=None)
g.dump_fwd_graph(f'test_fwd_graph_db.json')
g.dump_bwd_graph(f'test_bwd_graph_db.json')
g.dump_graph(f'test_fbwd_graph_db.json')
print("torch_database: 已完成fwd/bwd dump_graph...")
# db = (g._get_overall_database())
# json.dump(db,
#           open(args.model + 'db.json', 'w'),
#           indent=4)
