class NodeBuilder():
    def __init__(self, name, op):
        self.node = Node(name, op)

    def add_input_nodes(self, input_nodes):
        self.node.input_nodes = input_nodes

    def add_output_nodes(self, output_nodes):
        self.node.output_nodes = output_nodes

    def add_input_types(self, input_types):
        self.node.input_types = input_types

    def add_input_shapes(self, input_shapes):
        self.node.input_shapes = input_shapes

    def add_output_types(self, output_types):
        self.node.output_types = output_types

    def add_output_shapes(self, output_shapes):
        self.node.output_shapes = output_shapes

    def add_weight_type(self, weight_type):
        self.node.weight_type = weight_type

    def add_weight_shape(self, weight_shape):
        self.node.weight_shape = weight_shape

    def add_bias_type(self, bias_type):
        self.node.bias_type = bias_type

    def add_bias_shape(self, bias_shape):
        self.node.bias_shape = bias_shape

    def add_attrs(self, attrs):
        self.node.attrs = attrs

    def assemble_node(self):
        return self.node

class NodeEngineer(object):
    def __init__(self):
        self.builder = None

    def construct_node(self, name, op,
                       input_nodes = None,
                       output_nodes = None,
                       input_types = None,
                       input_shapes = None,
                       output_types = None,
                       output_shapes = None,
                       weight_type = None,
                       weight_shape = None,
                       bias_type = None,
                       bias_shape = None,
                       attrs = None
                       ):
        self.builder = NodeBuilder(name, op)
        if not input_nodes is None:
            self.builder.add_input_nodes(input_nodes)
        if not output_nodes is None:
            self.builder.add_output_nodes(output_nodes)
        if not input_types is None:
            self.builder.add_input_types(input_types)
        if not input_shapes is None:
            self.builder.add_input_shapes(input_shapes)
        if not output_types is None:
            self.builder.add_output_types(output_types)
        if not output_shapes is None:
            self.builder.add_output_shapes(output_shapes)
        if not weight_type is None:
            self.builder.add_weight_type(weight_type)
        if not weight_shape is None:
            self.builder.add_weight_shape(weight_shape)
        if not bias_type is None:
            self.builder.add_bias_type(bias_type)
        if not bias_shape is None:
            self.builder.add_bias_shape(bias_shape)
        if not attrs is None:
            self.builder.add_attrs(attrs)
        return self.builder.assemble_node()


class Node:
    def __init__(self, name: str, op: str):
        self.name = name 
        self.op = op  

    def __repr__(self):
        return str(self.to_json())

    def to_json(self):
        return self.__dict__
