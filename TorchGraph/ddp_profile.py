import json
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# from transformers import gpt2 #BertModel, BertConfig

# config = BertConfig(
#     hidden_size=1024,
#     num_hidden_layers=24,
#     num_attention_heads=16,
#     intermediate_size=4096
# )

### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--repeat", default=50, type=int)
parser.add_argument("--batchsize", default=16, type=int)
parser.add_argument('--model', type=str, default='gpt2',
                    help='model to benchmark')
parser.add_argument('--path', type=str, default='DDP.json',
                    help='path')
parser.add_argument('--bucket_cap_mb', type=int, default=25,
                    help='ddp bucket_cap_mb')
parser.add_argument('--type', type=str, default='CV',
                    help='model types')
FLAGS = parser.parse_args()
# local_rank = FLAGS.local_rank

print("script start")


local_rank = int(os.environ['LOCAL_RANK'])
bucket_cap_mb = FLAGS.bucket_cap_mb
# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
# FLAGS.type = "CV"
# FLAGS.type = "NLP"
# module = models.resnet152().to(local_rank)
# # DDP: 构造DDP model
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

from torchvision import models
import transformer

if FLAGS.type == 'CV':
    module = getattr(models, FLAGS.model)().cuda()
    example = torch.rand(FLAGS.batchsize, 3, 224, 224).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

elif FLAGS.type == 'NLP':
    print(f"test model -> {FLAGS.model}")
    module = getattr(transformer, FLAGS.model)().cuda()
    example = (torch.LongTensor(FLAGS.batchsize,256).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

module = DDP(module, bucket_cap_mb=bucket_cap_mb)


def benchmark_step():
    # 记录前向传播开始时间
    start_forward = time.perf_counter()
    
    optimizer.zero_grad()
    output = module(example)
    
    # 记录前向传播结束时间
    end_forward = time.perf_counter()
    
    # 记录后向传播开始时间
    start_backward = time.perf_counter()
    
    if FLAGS.type == 'CV':
        output.backward(output)
    elif FLAGS.type == 'NLP':
        if 'pooler_output' in output.__dict__:
            output.pooler_output.backward(output.pooler_output)
        else:
            output.last_hidden_state.backward(output.last_hidden_state)

    # 记录后向传播结束时间
    end_backward = time.perf_counter()
    
    # 记录优化器步骤开始时间
    start_optimizer = time.perf_counter()
    
    optimizer.step()
    
    # 记录优化器步骤结束时间
    end_optimizer = time.perf_counter()
    
    # 打印每个步骤的耗时（毫秒）
    print(f"Forward pass time: {(end_forward - start_forward) * 1000:.6f} ms")
    print(f"Backward pass time: {(end_backward - start_backward) * 1000:.6f} ms")
    print(f"Optimizer step time: {(end_optimizer - start_optimizer) * 1000:.6f} ms")

for i in range(5):
    benchmark_step()
torch.cuda.synchronize()
ss = time.perf_counter()
for i in range(FLAGS.repeat):
    benchmark_step()
torch.cuda.synchronize()
ee = time.perf_counter()

if local_rank == 0:
    print((ee - ss)/FLAGS.repeat*1000)


"""
export CUDA_VISIBLE_DEVICES=7
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ddp_profile.py


export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ddp_profile.py --model resnet152 --type CV --batchsize 128

export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ddp_profile.py --model gpt2 --type NLP --batchsize 16
"""
