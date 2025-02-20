import json
import torch
import torchvision
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# from transformers import BertModel, BertConfig

# config = BertConfig(
#     hidden_size=1024,
#     num_hidden_layers=24,
#     num_attention_heads=16,
#     intermediate_size=4096
# )

### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
# from torchvision import models
# module = models.resnet152().to(local_rank)
# # DDP: 构造DDP model
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

from transformers import GPT2Model, GPT2Config
config = GPT2Config(
        n_embd=1280,
        n_layer=24,
        n_head=20,
    )

module = GPT2Model(config).to(local_rank)
example = (torch.LongTensor(4,512).random_() % 1000).to(local_rank)
optimizer = optim.SGD(module.parameters(), lr=0.01)

module = DDP(module, device_ids=[local_rank], output_device=local_rank)

y = module(example)[0]
test = torch.ones_like(y)
y.backward(test)

def benchmark_step():
    optimizer.zero_grad()
    output = module(example)[0]
    output.backward(test)
    optimizer.step()

ss = time.perf_counter()
for i in range(100):
    benchmark_step()
ee = time.perf_counter()

print("real", (ee - ss)/100)
