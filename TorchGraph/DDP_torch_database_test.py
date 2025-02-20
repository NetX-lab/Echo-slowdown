import json
import torch
import torchvision
from torch_database import TorchDatabase
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time

# from transformers import BertModel, BertConfig

# config = BertConfig(
#     hidden_size=1024,
#     num_hidden_layers=24,
#     num_attention_heads=16,
#     intermediate_size=4096
# )

module = torchvision.models.alexnet(pretrained=True).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)
example = torch.rand(1, 3, 224, 224).cuda()

timer = Timer(100, 'alexnet')
g = TorchDatabase(module, example, 'alexnet', timer, optimizer)

# y = module(example)
# test = torch.ones_like(y)
# y.backward(test)

# def benchmark_step():
#     optimizer.zero_grad()
#     output = module(example)
#     output.backward(test)
#     optimizer.step()

# ss = time.perf_counter()
# for i in range(100):
#     benchmark_step()
# ee = time.perf_counter()

# print("real", (ee - ss)/100)



db = (g._get_overall_database())
json.dump(db,
          open('db.json', 'w'),
          indent=4)