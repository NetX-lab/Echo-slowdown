import torch
import torch.fx
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional

class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    stride : Tuple[int]
    memory_format : Optional[torch.memory_format]

    # Quantization metadata
    is_quantized : bool
    qscheme : Optional[torch.qscheme]
    q_scale : Optional[float]
    q_zero_point : Optional[int]

def extract_tensor_metadata(result : torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qscheme = None
    q_scale = None
    q_zero_point = None

    if is_quantized:
        qscheme = result.qscheme()

        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            q_scale = result.q_scale()
            q_zero_point = result.q_zero_point()


    return TensorMetadata(
        shape, dtype, stride, memory_format, is_quantized, qscheme, q_scale, q_zero_point)


class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super(TwoLayerNet, self).__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.dtype, node.shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed

    """
    def run_node(self, n : Node) -> Any:
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)

'''
 ShapeProp

ShapeProp类继承自torch.fx.Interpreter,用于逐节点执行FX图,并记录每个节点执行结果的形状(shape)和类型(type)。
当您调用ShapeProp(self._symbolic_traced_module).propagate(example)时,它会针对每个节点执行以下操作：

    1. 使用run_node方法逐节点执行图中的节点。
    2. 对于每个节点执行的结果,extract_tensor_meta函数会被调用来提取和记录张量的元数据,如形状、数据类型、步长(stride)等。这些信息被存储在节点的meta属性中。



'''