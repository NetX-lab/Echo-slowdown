import torch
from DDP_graph import DDPGraph
import torch.optim as optim
import argparse
from example_model import SimpleModel
import os


parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
args = parser.parse_args()


model = SimpleModel(input_features=100, hidden_dim1=64, hidden_dim2=32, output_features=10).cuda()
example_input = torch.randn((1, 100), requires_grad=True, device="cuda")
optimizer = optim.SGD(model.parameters(), lr=0.01)


from torchvision import models
module = getattr(models, args.model)().cuda()
