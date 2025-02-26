import json
import torch
import torch.fx
from torch.fx.node import Node, map_aggregate, map_arg
from torch.fx.graph import Graph
from torch.fx import symbolic_trace
from transformers.modeling_fx_utils import symbolic_trace as transformers_symbolic_trace
from shape_prop import ShapeProp, TensorMetadata, extract_tensor_metadata
from typename import typename
import Node
from transformers import PreTrainedModel

class TorchGraph:
    """
    Visualize a torch model with torch.fx.Graph
    Basic usage:
        g = TorchGraph(module, 'resnet18')
        with open("a.svg", "w") as f:
            f.write(g.get__graph().create_svg())
    """

    def __init__(self, name: str, module: torch.nn.Module, example: torch.tensor, 
                 loss_fn: torch.nn.Module = None, example_label: torch.tensor = None,
                 optimizer: torch.optim = None,
                 additional = None): 
        self._module = module
        self._example = example
        self._example_label = example_label
        self._loss_fn = loss_fn
        self._name = name
        self._optimizer = optimizer
        self._additional = additional
        self._NodeEngineer = Node.NodeEngineer()
        
        self._graph_dict = {}
        self._grad_fn_list = []
        self._backward_op_dict = {}
        self._backward_graph_dict = {}
        self._forward_graph = []
        self._backward_graph = []

        self.symbolic_trace()
        self._create_forward_graph()
        self._create_backward_graph()
        self._build_optimizer()

    def map_(self, a, fn):
        if isinstance(a, tuple):
            for elem in a:
                for sub_item in self.map_(elem, fn):
                    yield sub_item
        elif isinstance(a, list):
            for elem in a:
                for sub_item in self.map_(elem, fn):
                    yield sub_item
        elif isinstance(a, dict):
            for k, v in a.items():
                for sub_item in self.map_(v, fn):
                    yield sub_item
        else:
            yield fn(a)

    def graph_manipulation(self, module, placeholder = None):
        
        output_node = [n for n in self._symbolic_traced_module.graph.nodes if n.target == "output"][0]

        self._symbolic_traced_module.add_submodule(module.__class__.__name__, module)
        with self._symbolic_traced_module.graph.inserting_before(output_node):
            new_module = \
                self._symbolic_traced_module.graph.call_module(module_name=module.__class__.__name__)
            
            previous_node = list(self.map_(output_node.args,  lambda x :x))[0]
            args = [previous_node]
            new_module.args = tuple(args)

        output_node.args = map_aggregate(output_node.args, lambda x : new_module if (x==previous_node) else x)

        if placeholder is not None:
            self._symbolic_traced_module.register_buffer(placeholder, self._example_label)
            with self._symbolic_traced_module.graph.inserting_before(new_module):
                new_placeholder = \
                    self._symbolic_traced_module.graph.get_attr(placeholder)

            args = list(new_module.args)
            args.append(new_placeholder)
            new_module.args = tuple(args)

        self._symbolic_traced_module.recompile()

    def symbolic_trace(self):

        if isinstance(self._module, PreTrainedModel):
            self._symbolic_traced_module = transformers_symbolic_trace(
                self._module,
                batch_size=self._example.shape[0],
                sequence_length=self._example.shape[1],
            )

        else:
            self._symbolic_traced_module = symbolic_trace(self._module)

        if not self._additional is None:
            self.graph_manipulation(self._additional)

        if not self._loss_fn is None and not self._example_label is None:
            self.graph_manipulation(self._loss_fn, placeholder='target')

        ShapeProp(self._symbolic_traced_module).propagate(self._example)

        self._GraphModule = torch.fx.GraphModule(
            self._symbolic_traced_module, self._symbolic_traced_module.graph)

    def _get_leaf_node(self, target: str) -> torch.nn.Module:
        py_obj = self._GraphModule
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
            else:
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
        elif node.op == "call_function":
            return node.kwargs
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

    def _register_hook(self, var):

        output_list = list(self.map_(var, lambda x:x))

        var = output_list[0]

        self._record_grad_fn(var.grad_fn)
        BFS_list = []
        BFS_list.append(var.grad_fn)

        while BFS_list:
            grad_fn = BFS_list[0]
            BFS_list.pop(0)

            grad_fn.register_hook(self._make_backward_hook(grad_fn))

            if hasattr(grad_fn, 'next_functions'):
                for u in grad_fn.next_functions:
                    if u[0] is not None and u[0] not in self._backward_graph_dict:
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
                self._graph_dict['output'].output_nodes.append(backward_node.name)
                backward_node.input_nodes.append('output')

            self._backward_graph.append(backward_node)
            self._graph_dict[backward_node.name] = backward_node

    def _make_forward_hook(self):
        def hook(module, input, output):
            if 'module' not in output.grad_fn.metadata:
                output.grad_fn.metadata['module'] = module
        return hook

    def _create_backward_graph(self):

        for m in self._GraphModule.modules():
            if not m._modules:
                m.register_forward_hook(self._make_forward_hook())  

        loss = self._GraphModule(self._example)
        self._register_hook(loss)

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
                item.input_nodes.append('Optimizer_zero')
                Optimizer_zero.output_nodes.append(item.name)
    
        for item in self._graph_dict.values():
            if item.op == 'AccumulateGrad':
                item.output_nodes.append('optimizer_step')
                Optimizer_step.input_nodes.append(item.name)

        self._graph_dict['optimizer_zero'] = Optimizer_zero
        self._graph_dict['optimizer_step'] = Optimizer_step
    
    def get_forward_graph(self):
        return self._forward_graph

    def get_output_json(self):
        return [item.to_json() for item in self._graph_dict.values()]

    def dump_graph(self, path):
        json.dump(self.get_output_json(),
                  open(path, 'w'),
                  indent=4)