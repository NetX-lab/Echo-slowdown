# import json
# import os
import torch
import torch.fx
from torch.fx.node import Node, map_aggregate
from torch.fx import symbolic_trace
from shape_prop import ShapeProp, TensorMetadata, extract_tensor_metadata
from typename import typename
import Node
from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
from torch_graph import TorchGraph

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_bucket_indices(model):
    if not isinstance(model, DDP):
        raise RecursionError("Input model must be DistributedDataParallel model")

    # Build tuple of (module, parameter) for all parameters that require grads.
    modules_and_parameters = [
        [
            (module, parameter)
            for module_name, module in model.module.named_modules()
            for parameter in [
                param
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in model.parameters_to_ignore
            ]
        ]
    ]
    
    # Deduplicate any parameters that might be shared across child modules.
    memo = set()
    modules_and_parameters = [
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        [(m, p) for m, p in replica_mps if p not in memo and not memo.add(p)]
        for replica_mps in modules_and_parameters
    ]

    # Build list of parameters.
    parameters = [
        list(parameter for _, parameter in replica)
        for replica in modules_and_parameters
    ]

    # Checks if a module will produce a sparse gradient.
    def produces_sparse_gradient(module):
        if isinstance(module, torch.nn.Embedding) or isinstance(
            module, torch.nn.EmbeddingBag
        ):
            return module.sparse
        return False

    # Build list of booleans indicating whether or not to expect sparse
    # gradients for the corresponding parameters.
    expect_sparse_gradient = [
        list(produces_sparse_gradient(module) for module, _ in replica)
        for replica in modules_and_parameters
    ]

    # 写法对应于 class DistributedDataParallel(Module)
    # dist._DEFAULT_FIRST_BUCKET_BYTES是第一个bucket的大小限制，用于存放最先定义的几个参数，
    # 以防它们的梯度被分配到更大的bucket中，从而增加了不必要的延迟。实验表明，将这个值设置为1MB是合理的。
    # self.bucket_bytes_cap是除了第一个bucket外其他buckets的大小上限。
    # dist._DEFAULT_FIRST_BUCKET_BYTES: 1048576, i.e, 1MB
    # model.bucket_bytes_cap: 26214400, i.e, 25MB
    bucket_indices, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            parameters[0],
            [dist._DEFAULT_FIRST_BUCKET_BYTES, model.bucket_bytes_cap],
            expect_sparse_gradient[0],
        )

    return parameters[0], list(reversed(bucket_indices))

class DDPGraph(TorchGraph):
    """
    Visualize a torch model with torch.fx.Graph
    Basic usage:
        g = TorchGraph(module, 'resnet18')
        with open("a.svg", "w") as f:
            f.write(g.get__graph().create_svg())
    """

    def __init__(self, module: torch.nn.Module, example: torch.tensor, optimizer: torch.optim, name: str, local_rank: str, args: str):
    
        # local_rank = 0
        # torch.cuda.set_device(local_rank)

        # torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        module = DDP(module, device_ids=[local_rank], output_device=local_rank, bucket_cap_mb=1)

        self._module = module.module
        self._example = example
        self._name = name
        self._optimizer = optimizer

        self._parameters, self._bucket_indices = get_bucket_indices(module)

        # eg. {140428837965712: 0, 140428837998896: 1}
        self._param_map = {id(param): index for index, param in enumerate(self._parameters)}

        self._parameter_to_bucket = [0 for _ in self._parameters]
        self._bucket_size = [0 for _ in self._bucket_indices]
        
        # 建立parameter到bucket的映射
        # 累加每个bucket中所有parameter的元素数量到_bucket_size[bucket_index]中,.nelement()返回parameter中元素总数
        # eg. self._bucket_indices = [[0, 1]] => bucket_index=0,bucket_indice=[0, 1]
        # len(self._parameters) -> 2
        # so self._parameter_to_bucket = [0, 0]，只有一个bucket
        for bucket_index, bucket_indice in enumerate(self._bucket_indices):
            for i in bucket_indice:
                self._parameter_to_bucket[i] = bucket_index
                self._bucket_size[bucket_index] += self._parameters[i].nelement()

        self._bucket_list = [[] for _ in self._bucket_indices]

        self._NodeEngineer = Node.NodeEngineer()

        if isinstance(self._module, PreTrainedModel):
            self._symbolic_traced_module = transformers_symbolic_trace(self._module)
            ShapeProp(self._symbolic_traced_module).propagate(example)
        else:
            self._symbolic_traced_module = symbolic_trace(self._module)
            ShapeProp(self._symbolic_traced_module).propagate(example)

        self._graph_dict = {}
        self._grad_fn_list = []
        self._backward_op_dict = {}
        self._backward_graph_dict = {}
        self._forward_graph = []
        self._backward_graph = []

        self._create_forward_graph()
        if not args.no_bw:
            self._create_backward_graph()
        if not args.no_ddp:
            self._create_DDP_graph()
        if not args.no_op:
            self._build_optimizer()

    def _make_backward_hook(self, node):
        def hook(inputs, outputs):
            if self._get_bp_node_op(node) not in self._backward_op_dict:
                self._backward_op_dict[self._get_bp_node_op(node)] = 0
            else:
                self._backward_op_dict[self._get_bp_node_op(node)] += 1
            self._backward_graph_dict[node]['name'] = \
                self._get_bp_node_op(node) +str(self._backward_op_dict[self._get_bp_node_op(node)])
            self._backward_graph_dict[node]['input_meta'] = \
                self._get_tensor_meta(outputs)
            self._backward_graph_dict[node]['output_meta'] = \
                self._get_tensor_meta(inputs)

            self._grad_fn_list.append(node)

            # self._bucket_list记录bw过程中的AccumulateGrad
            # eg. self._bucket_list: [['AccumulateGrad0', 'AccumulateGrad1']]
            # 该过程无法debug, 因为使用了hook方法
            if hasattr(node, 'variable'):
                # 即node对应的param在之前遍历的记录中
                if id(node.variable) in self._param_map:
                    # self._backward_graph_dict[node]['index'] = self._param_map[id(node.variable)]
                    index_ = self._param_map[id(node.variable)]
                    self._bucket_list[self._parameter_to_bucket[index_]].append(self._get_bp_node_name(node))
                else:
                    raise RuntimeError("find unseen parameters on bp graph")

        return hook

    def _create_DDP_graph(self):

        for index_, bucket in enumerate(self._bucket_list):
            pre_bucket_node = self._NodeEngineer.construct_node(
                name="ddp_pre_" + str(index_),
                op="ddp_pre",
                input_nodes=bucket,
                output_nodes=["ddp_Allreduce_" + str(index_)],
                input_types=[],
                input_shapes=[],
                output_types=[],
                output_shapes=[],
                attrs={'bucket_size':self._bucket_size[index_]}
            )
            
            bucket_node = self._NodeEngineer.construct_node(
                name="ddp_Allreduce_" + str(index_),
                op="ddp_Allreduce",
                input_nodes=["ddp_pre_" + str(index_)],
                output_nodes=[],
                input_types=[],
                input_shapes=[],
                output_types=[],
                output_shapes=[],
                attrs={'bucket_size':self._bucket_size[index_]}
            )

            for AccumulateGrad in bucket:
                # 为每个AccumulateGrad op添加output node(ddp_pre)
                self._graph_dict[AccumulateGrad].output_nodes.append(pre_bucket_node.name)

            # ddp op加入到graph中
            self._graph_dict[pre_bucket_node.name] = pre_bucket_node
            self._graph_dict[bucket_node.name] = bucket_node

            # ddp op之间的连接
            if index_ != 0:
                self._graph_dict["ddp_pre_" + str(index_)].input_nodes.append("ddp_Allreduce_" + str(index_-1))
                self._graph_dict["ddp_Allreduce_" + str(index_-1)].output_nodes.append("ddp_pre_" + str(index_))
