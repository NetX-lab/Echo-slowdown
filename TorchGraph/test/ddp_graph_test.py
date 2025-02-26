'''
export CUDA_VISIBLE_DEVICES=0,1
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ../trace_file/ddp_test \
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 ddp_graph_test.py --no_op

export CUDA_VISIBLE_DEVICES=1
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ./trace_data/$output_file_name \
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ddp_graph_test.py --no_op

'''
# import json
import torch
from DDP_graph import DDPGraph
import torch.optim as optim
import argparse
from example_model import SimpleModel
import os

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--no_bw', action='store_true', 
                    help='Skip the execution of self._create_backward_graph()')
parser.add_argument('--no_ddp', action='store_true', 
                    help='Skip the execution of self._create_DDP_graph()')
parser.add_argument('--no_op', action='store_true', 
                    help='Skip the execution of self._build_optimizer()')
parser.add_argument("--save_path", default="../trace_file/DDP_test-no-op-3layers.json", type=str)

args = parser.parse_args()

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = SimpleModel().cuda()
example_input = torch.randn(1, 512).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)


g = DDPGraph(model, example_input, optimizer, 'example-net', local_rank, args)
# g.dump_graph('DDP_test-no-op-3layers.json')
g.dump_graph(args.save_path)