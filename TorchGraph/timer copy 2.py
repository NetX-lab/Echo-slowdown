import time
import sys
import torch
import statistics 
import torch.autograd.profiler as torch_profiler
sys.setrecursionlimit(1500)

class Timer():

    def __init__(self, profiling_steps: int, name: str):
        self.profiling_steps = 1
        self.name = name
        self.database = dict()
        self.variance = dict()
        self.grad_fn_list = []
        self.grad_fn_input_list = []
        self.backward_op_dict = dict()

    def _init_database(self):
        self.database = dict()
        self.variance = dict()

    def _bp_profiling(self):
        for var_name, outputs in zip(self.grad_fn_list, self.grad_fn_input_list):
            name = var_name['name']
            var = var_name['var']
            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                with torch_profiler.profile(use_cuda=True) as prof:
                    var(*outputs)
                cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
                self.database[name] = cuda_total / 1e6
                self.variance[name] = 0
                print(name)

    def _get_bp_node_op(self, var):
        return type(var).__name__

    def _make_hook(self, var):
        def hook(inputs, outputs):
            if self._get_bp_node_op(var) not in self.backward_op_dict:
                self.backward_op_dict[self._get_bp_node_op(var)] = 0
            else:
                self.backward_op_dict[self._get_bp_node_op(var)] += 1

            name = self._get_bp_node_op(var) + str(self.backward_op_dict[self._get_bp_node_op(var)])
            print(name)

            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                with torch_profiler.profile(use_cuda=True) as prof:
                    var(*outputs)
                cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
                print(name)
                self.database[name] = cuda_total / 1e6
                self.variance[name] = 0
            
            # self.grad_fn_list.append({'var':var, 'name':self._get_bp_node_op(var) +str(self.backward_op_dict[self._get_bp_node_op(var)])})
            # self.grad_fn_input_list.append(outputs)

        return hook

    def _empty_hook(self, var):
        def hook(inputs, outputs):
            pass
        return hook

    def _call_function(self, function, node, args, kwargs):
        with torch_profiler.profile(use_cuda=True) as prof:
            output = function(node.target, args, kwargs)
        cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
        self.database[node.name] = cuda_total / 1e6
        self.variance[node.name] = 0
        return output


    def _call_function_once(self, function, node, args, kwargs):
        output = function(node.target, args, kwargs)
        self.database[node.name] = 0
        self.variance[node.name] = 0
        return output


    def _call_optimizer(self, function, name):
        
        with torch_profiler.profile(use_cuda=True) as prof:
            function()
        cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
        self.database[name] = cuda_total / 1e6
        self.variance[name] = 0
    
    def _get_database(self):
        return self.database

    def _get_variance(self):
        return self.variance


def make_dot(var, params, hook):
    """ Produces Graphviz representation of PyTorch autograd graph.
    
    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.
    
    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    
    param_map = {id(v): k for k, v in params}

    seen = set()
    
    def add_nodes(var):
        if var not in seen:
            
            node_id = str(id(var))
            
            var.register_hook(hook(var))
            # print(var.name())
            # if(var.name() == "CudnnConvolutionBackward"):
            #     print(var.getAttribute("padding"))
            seen.add(var)
            
            if hasattr(var, 'next_functions'):
                # print(var.name(), var.next_functions)
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
                        
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t)

    add_nodes(var[0].grad_fn)
