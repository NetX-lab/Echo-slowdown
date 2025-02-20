"""
_create_forward_graph()
处理由symbolic_trace()生成的计算图信息。
具体来说,它遍历self._symbolic_traced_module.graph.nodes,即symbolic_trace()产生的图中的所有节点,并为每个节点创建一个详细的表示,这包括：

    1. 为每个节点提取和记录更多的运行时信息,例如输入输出张量的形状和数据类型。这是通过之前调用的ShapeProp类在节点的meta属性中添加的信息来实现的。
    2. 为每个节点标识出其操作类型、关联的模块(如果是调用模块的操作）、输入节点和输出节点等。这样不仅包括了节点的静态结构,还包括了更多动态执行过程中的详细信息。
    3. 构建一个节点工程对象,即通过_NodeEngineer实例创建每个节点的详细表示。这包括节点的名字、操作类型、输入输出节点、输入输出张量的形状和类型、权重和偏置(如果有的话)的数据类型和形状等。

_create_backward_graph()
遍历grad_fn(生成该张量的函数)和它们的next_functions,逐步构建起整个模型的梯度计算图。这个过程中的关键步骤包括：

    1. 注册forward hook以收集每个操作输出张量的grad_fn,以及它们对应的模块信息。
    2. 使用输出张量调用backward(),触发后向传播过程。
    3. 在后向传播过程中,通过注册的hook记录每个梯度函数grad_fn的输入输出信息,包括张量的形状和数据类型。

"""

import json
import torch
import torch.fx
from torch.fx.node import Node, map_aggregate
from torch.fx import symbolic_trace
# from .shape_prop import ShapeProp, TensorMetadata, extract_tensor_metadata
from shape_prop import ShapeProp, TensorMetadata, extract_tensor_metadata
from typename import typename
import Node
import sys
from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
sys.setrecursionlimit(1500)


def forward_hook_fn(module, input, output):
    if 'module' not in output.grad_fn.metadata:
        output.grad_fn.metadata['module'] = module

class TorchGraph:
    def __init__(self, module: torch.nn.Module, example: torch.tensor, optimizer: torch.optim, name: str):
        self._module = module
        self._example = example
        self._name = name
        self._optimizer = optimizer
        self._NodeEngineer = Node.NodeEngineer()
        if isinstance(module, PreTrainedModel):
            from transformers.utils.fx import symbolic_trace
            # self._symbolic_traced_module = transformers_symbolic_trace(module) # , concrete_args={'position_ids':example[1]}
            self._symbolic_traced_module = symbolic_trace(module)
            ShapeProp(self._symbolic_traced_module).propagate(example)
        else:
            from transformers.utils.fx import symbolic_trace
            # self._symbolic_traced_module = symbolic_trace(module)
            self._symbolic_traced_module = symbolic_trace(module)
            ShapeProp(self._symbolic_traced_module).propagate(example)

        self._graph_dict = {}
        self._fwd_graph_dict = {}
        self._bwd_graph_dict = {} # recored node for json file, different from _backward_graph_dict
        self._optim_graph_dict = {}
        self._grad_fn_list = []
        self._backward_op_dict = {}
        self._backward_graph_dict = {}
        self._forward_graph = []
        self._backward_graph = []
        self._create_forward_graph()
        self._create_backward_graph()
        self._build_optimizer()

    def _get_leaf_node(self, target: str) -> torch.nn.Module:
        py_obj = self._module
        atoms = target.split(".")
        for atom in atoms:
            if not hasattr(py_obj, atom):
                raise RuntimeError(
                    str(py_obj) + " does not have attribute " + atom + "!"
                )
            py_obj = getattr(py_obj, atom)
        return py_obj

    def _get_node_name(self, node: torch.fx.node.Node):
        return node.name

    def _get_node_op(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            return typename(self._get_leaf_node(node.target))
        else:
            return typename(node.target)

    def _get_node_input_nodes(self, node: torch.fx.node.Node):
        return [node.name for node in node._input_nodes]

    def _get_node_output_nodes(self, node: torch.fx.node.Node):
        return [node.name for node in node.users]

    def _insert_meta_obj(self, meta_obj):
        if isinstance(meta_obj, TensorMetadata):
            self._Metadata_list.append(meta_obj)
        else:
            if isinstance(meta_obj, dict):
                for obj in meta_obj.values():
                    self._insert_meta_obj(obj)
            elif isinstance(meta_obj, list):
                for obj in meta_obj:
                    self._insert_meta_obj(obj)

    def _get_node_input_tensor_metadata(self, node: torch.fx.node.Node):
        self._Metadata_list = []
        for input_node in node._input_nodes.keys():
            meta_obj = input_node.meta.get('tensor_meta')
            self._insert_meta_obj(meta_obj)
        return self._Metadata_list

    def _get_node_tensor_metadata(self, node: torch.fx.node.Node):
        self._Metadata_list = []
        meta_obj = node.meta.get('tensor_meta')
        self._insert_meta_obj(meta_obj)
        return self._Metadata_list

    def _get_node_output_tensor_metadata(self, node: torch.fx.node.Node):
        self._Metadata_list = []
        for input_node in node.users.keys():
            meta_obj = input_node.meta.get('tensor_meta')
            self._insert_meta_obj(meta_obj)
        return self._Metadata_list

    def _get_node_input_types(self, node: torch.fx.node.Node):
        return [str(Metadata.dtype) for Metadata in self._get_node_input_tensor_metadata(node)]

    def _get_node_input_shapes(self, node: torch.fx.node.Node):
        return [Metadata.shape for Metadata in self._get_node_input_tensor_metadata(node)]

    def _get_node_output_types(self, node: torch.fx.node.Node):
        return [str(Metadata.dtype) for Metadata in self._get_node_tensor_metadata(node)]

    def _get_node_output_shapes(self, node: torch.fx.node.Node):
        return [Metadata.shape for Metadata in self._get_node_tensor_metadata(node)]

    def _get_node_weight_type(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            leaf_module = self._get_leaf_node(node.target)
            if hasattr(leaf_module, "weight") and leaf_module.weight is not None:
                return str(leaf_module.weight.dtype)
            else:
                return None
        else:
            return None

    def _get_node_weight_shape(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            leaf_module = self._get_leaf_node(node.target)
            if hasattr(leaf_module, "weight") and leaf_module.weight is not None:
                return leaf_module.weight.size()
            else:
                return None
        else:
            return None

    def _get_node_bias_type(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            leaf_module = self._get_leaf_node(node.target)
            if hasattr(leaf_module, "bias") and leaf_module.bias is not None:
                return str(leaf_module.bias.dtype)
            else:
                return None
        else:
            return None

    def _get_node_bias_shape(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            leaf_module = self._get_leaf_node(node.target)
            if hasattr(leaf_module, "bias") and leaf_module.bias is not None:
                return leaf_module.bias.size()
            else:
                return None
        else:
            return None

    def _get_node_attr(self, node: torch.fx.node.Node):
        if node.op == "call_module":
            leaf_module = self._get_leaf_node(node.target)
            attr = {}
            if hasattr(leaf_module, "__constants__"):
                for c in leaf_module.__constants__:
                    attr[c] = getattr(leaf_module, c) 
            return attr
        else:
            return None

    def _create_forward_node(self, node: torch.fx.node.Node, op: str):
        return self._NodeEngineer.construct_node(
            name=self._get_node_name(node),
            op=self._get_node_op(node),
            input_nodes=self._get_node_input_nodes(node),
            output_nodes=self._get_node_output_nodes(node),
            input_types=self._get_node_input_types(node),
            input_shapes=self._get_node_input_shapes(node),
            output_types=self._get_node_output_types(node),
            output_shapes=self._get_node_output_shapes(node),
            weight_type=self._get_node_weight_type(node),
            weight_shape=self._get_node_weight_shape(node),
            bias_type=self._get_node_bias_type(node),
            bias_shape=self._get_node_bias_shape(node),
            attrs=self._get_node_attr(node)
        )

    def _create_forward_graph(self):
        for node in self._symbolic_traced_module.graph.nodes:
            forward_node = self._create_forward_node(node, self._get_node_op(node))
            self._forward_graph.append(forward_node)
            self._graph_dict[forward_node.name] = forward_node
            self._fwd_graph_dict[forward_node.name] = forward_node

    def _get_bp_node_attr(self, node: torch.fx.node.Node):
        if 'module' in node.metadata:
            attr = {}
            if hasattr(node.metadata['module'], "__constants__"):
                for c in node.metadata['module'].__constants__:
                    attr[c] = getattr(node.metadata['module'], c) 
            return attr
        else:
            return None

    def _get_bp_node_name(self, node):
        return self._backward_graph_dict[node]['name']

    def _get_bp_node_op(self, node):
        return type(node).__name__

    def _get_bp_node_input_nodes(self, node):
        return [self._backward_graph_dict[node]['name'] \
                for node in self._backward_graph_dict[node]['input_nodes']]

    def _get_bp_node_output_nodes(self, node):
        return [self._backward_graph_dict[node]['name'] \
                for node in self._backward_graph_dict[node]['output_nodes']]

    def _get_tensor_meta(self, result):
        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                return extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        return meta

    def _get_bp_node_input_types(self, node):
        return [str(Metadata.dtype) if Metadata else Metadata \
                for Metadata in self._backward_graph_dict[node]['input_meta']]
    
    def _get_bp_node_input_shapes(self, node):
        return [(Metadata.shape) if Metadata else Metadata \
                for Metadata in self._backward_graph_dict[node]['input_meta']]

    def _get_bp_node_output_types(self, node):
        return [str(Metadata.dtype) if Metadata else Metadata \
                for Metadata in self._backward_graph_dict[node]['output_meta']]

    def _get_bp_node_output_shapes(self, node):
        return [(Metadata.shape) if Metadata else Metadata \
                for Metadata in self._backward_graph_dict[node]['output_meta']]


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
        return hook

    def _record_grad_fn(self, grad_fn):
        self._backward_graph_dict[grad_fn] = {
            'input_nodes':[],
            'output_nodes':[]
        }


    def _insert_tensor_obj(self, tensor_obj):
        if isinstance(tensor_obj, torch.Tensor):
            self._output_list.append(tensor_obj)
        else:
            if isinstance(tensor_obj, dict):
                for obj in tensor_obj.values():
                    self._insert_tensor_obj(obj)
            else:
                for obj in tensor_obj:
                    self._insert_tensor_obj(obj)

    def _register_hook(self, var):

        self._output_list = []
        self._insert_tensor_obj(var) 
        BFS_list = []

        # for output in self._output_list:
            # print(output.grad_fn)
        var = self._output_list[0]
        self._record_grad_fn(var.grad_fn)
        # self._graph_dict['output'].output_nodes.append(self._get_bp_node_name(output.grad_fn))
        # self._backward_graph_dict[output.grad_fn]['input_nodes'].append('output')

        BFS_list.append(var.grad_fn)

        while BFS_list:
            grad_fn = BFS_list[0]
            BFS_list.pop(0)

            grad_fn.register_hook(self._make_backward_hook(grad_fn))

            if hasattr(grad_fn, 'next_functions'):
                for u in grad_fn.next_functions:
                    if u[0] is not None:
                        if u[0] not in self._backward_graph_dict:
                            BFS_list.append(u[0])

                            self._record_grad_fn(u[0])

                        self._backward_graph_dict[grad_fn]['output_nodes'].append(u[0])
                        self._backward_graph_dict[u[0]]['input_nodes'].append(grad_fn)
        try:
            var.backward()
        except:
            var.backward(var)

        for node in self._grad_fn_list:
            
            backward_node = self._NodeEngineer.construct_node(
                name=self._get_bp_node_name(node),
                op=self._get_bp_node_op(node),
                input_nodes=self._get_bp_node_input_nodes(node),
                output_nodes=self._get_bp_node_output_nodes(node),
                input_types=self._get_bp_node_input_types(node),
                input_shapes=self._get_bp_node_input_shapes(node),
                output_types=self._get_bp_node_output_types(node),
                output_shapes=self._get_bp_node_output_shapes(node),
                attrs=self._get_bp_node_attr(node),
            )

            if node == var.grad_fn:
                # Note: bwd_graph need additional ouput node (last node from fwd graph) as input node
                # self._bwd_graph_dict['output'] = self._graph_dict['output']
                # self._bwd_graph_dict['output'].output_nodes.append(backward_node.name)

                self._graph_dict['output'].output_nodes.append(backward_node.name)
                # print(f"{self._graph_dict['output']}")
                # print(f"{self._graph_dict['output'].output_nodes}")
                backward_node.input_nodes.append('output')

            self._backward_graph.append(backward_node)
            self._graph_dict[backward_node.name] = backward_node
            # self._bwd_graph_dict[backward_node.name] = backward_node

    def _make_forward_hook(self):
        def hook(module, input, output):
            try:
                if 'module' not in output.grad_fn.metadata:
                    output.grad_fn.metadata['module'] = module
            except:
                pass
                # print(f"output:{output}")
                # print(f"output.grad_fn:{output.grad_fn}")
        return hook

    def _create_backward_graph(self):
        # access module attributes for backward grad_fn
        for m in self._module.modules():
            if not m._modules:
                m.register_forward_hook(self._make_forward_hook())

        # create_backward_graph by hook
        output_example = self._module(self._example)
        # print(f"Type of output_example: {type(output_example)}")
        if isinstance(self._module, PreTrainedModel):
            if isinstance(output_example, tuple):
                output_example = output_example[0]  # adjust this as needed
            if hasattr(output_example, 'pooler_output'):
                output_example = output_example.pooler_output
            elif hasattr(output_example, 'last_hidden_state'):
                output_example = output_example.last_hidden_state
        # if isinstance(self._module, PreTrainedModel):
        #     if 'pooler_output' in output_example.__dict__:
        #         output_example = output_example.pooler_output
        #     else:
        #         output_example = output_example.last_hidden_state
        # 递归地为每个梯度函数（grad_fn）注册一个钩子,
        # 这些钩子的目的是在后向传播时捕获每个操作的输入和输出元数据（如形状和类型），
        # 以及操作本身的信息，这些信息用于构建后向计算图的详细表示。
        self._register_hook(output_example)

    def _build_optimizer(self):
        self._optimizer_params_type = []
        self._optimizer_params_shape = []
        for group in self._optimizer.param_groups:
             for p in group['params']:
                self._optimizer_params_type.append(str(p.dtype))
                self._optimizer_params_shape.append((p.shape))

        Optimizer_zero = self._NodeEngineer.construct_node(
            name="optimizer_zero",
            op="optimizer_zero",
            input_nodes=[],
            output_nodes=[],
            input_types=self._optimizer_params_type,
            input_shapes=self._optimizer_params_shape,
            output_types=[],
            output_shapes=[],
            attrs=self._optimizer.defaults
        )

        Optimizer_step = self._NodeEngineer.construct_node(
            name="optimizer_step",
            op="optimizer_step",
            input_nodes=[],
            output_nodes=[],
            input_types=self._optimizer_params_type,
            input_shapes=self._optimizer_params_shape,
            output_types=[],
            output_shapes=[],
            attrs=self._optimizer.defaults
        )

        for item in self._graph_dict.values():
            if item.input_nodes == []:
                # item.input_nodes.append('optimizer_zero')
                Optimizer_zero.output_nodes.append(item.name)
    
        for item in self._graph_dict.values():
            if item.output_nodes == []:
                # item.output_nodes.append('optimizer_step')
                Optimizer_step.input_nodes.append(item.name)

        self._optim_graph_dict['optimizer_zero'] = Optimizer_zero
        self._optim_graph_dict['optimizer_step'] = Optimizer_step
        
        # graph dict中暂不包含optim的算子
        # self._graph_dict['optimizer_zero'] = Optimizer_zero
        # self._graph_dict['optimizer_step'] = Optimizer_step
    
    def get_forward_graph(self):
        return self._forward_graph

    def get_backward_graph(self):
        return self._backward_graph

    def get_output_json(self):
        return [item.to_json() for item in self._graph_dict.values()]

    def get_fwd_output_json(self):
        del self._fwd_graph_dict['output']
        return [item.to_json() for item in self._fwd_graph_dict.values()]

    def get_bwd_output_json(self):
        self._bwd_graph_dict = {
            key: value
            for key, value in self._graph_dict.items()
            if key not in self._fwd_graph_dict
        }

        return [item.to_json() for item in self._bwd_graph_dict.values()]

    def get_optim_output_json(self):
        return [item.to_json() for item in self._optim_graph_dict.values()]

    def dump_graph(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_output_json(), file, indent=4)

    def dump_fwd_graph(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_fwd_output_json(), file, indent=4)

    def dump_bwd_graph(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_bwd_output_json(), file, indent=4)

    def dump_optim_graph(self, path):
        with open(path, 'w') as file:
            json.dump(self.get_optim_output_json(), file, indent=4)