"""

python ddp_graph_database_test.py --model vgg19 --type CV --batchsize 256
export CUDA_VISIBLE_DEVICES=7
python ddp_graph_database_test.py --model resnet152 --type CV --batchsize 128
export CUDA_VISIBLE_DEVICES=1
python ddp_graph_database_test.py --model gpt2 --type NLP --batchsize 16

"""

import json
import torch
# import torchvision
from torch_database import TorchDatabase
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time
# from example_model import SimpleModel
import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet152',
                    help='model to benchmark')
parser.add_argument('--type', type=str, default='CV',
                    help='model types')
parser.add_argument("--batchsize", default=32, type=int)
args = parser.parse_args()

# 测量 VGG19 时，batchsize为 256，type为 CV
# 测量 resnet152 时，batchsize为128，type为 CV
# 测量 gpt2 时，batchsize为 16，type为 NLP
from torchvision import models
import transformer
if args.type == 'CV':
    module = getattr(models, args.model)().cuda()
    example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
elif args.type == 'NLP':
    module = getattr(transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)


use_ncu = True
timer = Timer(50, args.model,use_ncu=use_ncu)
g = TorchDatabase(module, example, args.model, timer, optimizer)
print(f"Finished profiling.")


db = (g._get_overall_database())
json.dump(db,
          open(args.model + 'db.json', 'w'),
          indent=4)