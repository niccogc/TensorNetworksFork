import torch
import torch.nn as nn
from tensor.network import TensorNetwork
from tensor.node import TensorNode

class TensorNetworkLayer(nn.Module):
    def __init__(self, tensor_network: TensorNetwork, labels=None):
        """Initializes a TensorNetworkLayer."""
        super(TensorNetworkLayer, self).__init__()
        self.tensor_network = tensor_network
        self.labels = labels if labels is not None else tensor_network.output_labels

    def cuda(self, *args, **kwargs):
        """Moves the layer to the GPU."""
        self.tensor_network.cuda()
        return super(TensorNetworkLayer, self).cuda(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Moves the layer to the given device and dtype."""
        self.tensor_network.to(*args, **kwargs)
        return super(TensorNetworkLayer, self).to(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        """Moves the layer to the CPU."""
        self.tensor_network.to('cpu')
        return super(TensorNetworkLayer, self).cpu(*args, **kwargs)

    def forward(self, x):
        """Forward pass of the layer."""
        tn_out = self.tensor_network.forward(x)
        if self.labels is not None:
            tn_out.permute_first(*self.labels)
        return tn_out.tensor

class TensorTrainLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, input_features, output_shape=tuple(), ring=False, squeeze=True):
        """Initializes a TensorTrainLayer."""
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.input_features = input_features
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring

        # Create input nodes
        self.x_nodes = []
        for i in range(1, num_carriages+1):
            x_node = TensorNode((1, input_features), ['s', 'p'], name=f"X{i}")
            self.x_nodes.append(x_node)

        # Create main nodes
        self.nodes = []
        self.labels = ['s']
        for i in range(1, num_carriages+1):
            if i-1 < len(self.output_shape):
                up = self.output_shape[i-1]
                up_label = f'c{i}'
                self.labels.append(up_label)
            else:
                up = 1
                up_label = 'c'
            down = input_features
            left_label = 'rr' if ring and i == 1 else f'r{i}'
            right_label = 'rr' if ring and i == num_carriages else f'r{i+1}'
            if ring:
                left = bond_dim
                right = bond_dim
            else:
                if i == 1:
                    left = 1
                    right = bond_dim
                elif i == num_carriages:
                    left = bond_dim
                    right = 1
                else:
                    left = bond_dim
                    right = bond_dim
            node = TensorNode((left, up, down, right), [left_label, up_label, 'p', right_label], l=left_label, r=right_label, name=f"A{i}")
            if i > 1:
                self.nodes[-1].connect(node, left_label, priority=1)
            if ring and i == num_carriages:
                node.connect(self.nodes[0], right_label, priority=0)
            node.connect(self.x_nodes[i-1], 'p', priority=2)
            self.nodes.append(node)
        
        # Squeeze singleton dimensions
        if squeeze:
            for node in self.nodes:
                node.squeeze(self.labels)
        # Create a TensorNetwork
        tensor_network = TensorNetwork(self.x_nodes, self.nodes, output_labels=self.labels)
        super(TensorTrainLayer, self).__init__(tensor_network)

class TensorOperatorLayer(TensorNetworkLayer):
    def __init__(self, operator, input_features, bond_dim, num_carriages, output_shape=1, ring=False, left=None, right=None):
        """Initializes a TensorOperatorLayer."""
        self.operator = operator
        self.input_features = input_features
        self.bond_dim = bond_dim
        self.num_carriages = num_carriages
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring

        self.x_nodes = []
        self.op_nodes = []
        self.nodes = []
        self.output_labels = ('s',)

        # Create input nodes
        for i in range(num_carriages):
            x_node = TensorNode((1, input_features), ('s', f'd{i}'), name=f'X{i}')
            self.x_nodes.append(x_node)

        # Create operator nodes
        for i in range(num_carriages):
            # Select operator tensor
            if isinstance(operator, list) or isinstance(operator, tuple):
                O = operator[i]
            elif ring:
                O = operator
            elif i == 0 and left is not None:
                O = left
            elif i == num_carriages - 1 and right is not None:
                O = right
            elif i == 0:
                O = operator[:1]
            elif i == num_carriages - 1:
                O = operator[..., -1:]
            else:
                O = operator
            left_label = 'br' if ring and i == 0 else f'b{i}'
            right_label = 'br' if ring and i == num_carriages - 1 else f'b{i+1}'
            op_node = TensorNode(O, (left_label, f'u{i}', f'd{i}', right_label), l=left_label, r=right_label, name=f"O{i}")
            op_node.connect(self.x_nodes[i], f'd{i}')
            if i > 0:
                self.op_nodes[-1].connect(op_node, left_label)
            if ring and i == num_carriages - 1:
                op_node.connect(self.op_nodes[0], right_label)
            self.op_nodes.append(op_node)

        # Create main nodes
        for i in range(num_carriages):
            left_label = 'rr' if ring and i == 0 else f'r{i}'
            right_label = 'rr' if ring and i == num_carriages - 1 else f'r{i+1}'

            if ring:
                left_dim = right_dim = bond_dim
            else:
                if i == 0:
                    left_dim, right_dim = 1, bond_dim
                elif i == num_carriages - 1:
                    left_dim, right_dim = bond_dim, 1
                else:
                    left_dim = right_dim = bond_dim
            if i < len(self.output_shape):
                up_dim = self.output_shape[i]
                self.output_labels = self.output_labels + (f'c{i}',)
            else:
                up_dim = 1
                if i == 0:
                    self.output_labels = self.output_labels + ('c0',)
            block = torch.randn((left_dim, up_dim, input_features, right_dim))
            node = TensorNode(block, (left_label, f'c{i}', f'u{i}', right_label), l=left_label, r=right_label, name=f"A{i}")
            node.connect(self.op_nodes[i], f'u{i}')
            if i > 0:
                self.nodes[-1].connect(node, left_label)
            self.nodes.append(node)

        for node in self.nodes:
            node.squeeze(('c0',))
        for node in self.op_nodes:
            node.squeeze()

        if ring:
            self.nodes[-1].connect(self.nodes[0], 'rr')

        tensor_network = TensorNetwork(self.x_nodes, self.nodes, output_labels=self.output_labels)
        super(TensorOperatorLayer, self).__init__(tensor_network, labels=self.output_labels)


class TensorConvolutionTrainLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, num_patches, patch_pixels, output_shape, ring=False):
        """Initializes a TensorConvolutionTrainLayer."""
        if ring:
            raise NotImplementedError("Ring structure is not implemented for TensorConvolutionTrainLayer.")
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.num_patches = num_patches
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring

        self.output_labels = ('s',)

        # Create nodes
        x_nodes = []
        conv_blocks = []
        train_blocks = []
        for i in range(1, num_carriages+1):
            x_node = TensorNode((1, num_patches, patch_pixels), ['s', 'patches', 'patch_pixels'], name=f"X{i}")
            conv_block = TensorNode((patch_pixels,), ['patch_pixels'], name=f"C{i}")
            train_block = TensorNode((bond_dim if i != 1 else 1, self.output_shape[i-1] if i <= len(self.output_shape) else 1, num_patches, bond_dim if i != num_carriages else 1), [f'r{i}', f'c{i}', 'patches', f'r{i+1}'], l=f'r{i}', r=f'r{i+1}', name=f"A{i}")
            x_nodes.append(x_node)
            conv_blocks.append(conv_block)
            train_blocks.append(train_block)
            if i < len(self.output_shape)-1:
                self.output_labels = self.output_labels + (f'c{i}',)
        
        self.nodes = []
        for xn, cb, tb in zip(x_nodes, conv_blocks, train_blocks):
            cb.connect(xn, 'patch_pixels')
            xn.connect(tb, 'patches')
            self.nodes.append(cb)
            self.nodes.append(tb)
        
        for i in range(1, num_carriages):
            train_blocks[i-1].connect(train_blocks[i], f'r{i+1}')

        #for n in train_blocks:
            #n.squeeze()
        
        # Create a TensorNetwork
        self.x_nodes = x_nodes
        self.labels = self.output_labels
        tensor_network = TensorNetwork(x_nodes, train_blocks, output_labels=self.labels)
        super(TensorConvolutionTrainLayer, self).__init__(tensor_network)