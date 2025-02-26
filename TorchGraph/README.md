# TorchGraph

TorchGraph is a visualization tool for dumping PyTorch model description by symbolic-traced method.

## Install

### Download directly from the repo

## Run 

### Get baseline 

get a baseline by running a profiler script
```
python3 real.py --model=resnet50 
```

get a distributed baseline by running a profiler script
```
python3 -m torch.distributed.launch --nproc_per_node 2 DDP_real.py --model=resnet50 
```
The baseline results are shown on the system log.

### Get model description

get model description by Torchgraph
```
python3 torch_graph_test.py --model=resnet50 
```

get distributed baseline by model description by Torchgraph
```
python3 -m torch.distributed.launch --nproc_per_node 1 DDP_test.py --model=resnet50 
```
They dump the graph file locally.

### Get profiler database

get profiler database by torch_database.py
```
python3 torch_database_test.py --model=resnet50 
```
It dumps the database file locally.


