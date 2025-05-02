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
        self.parametrized = False
    
    def parametrize(self):
        self.tensor_params = nn.ParameterList()
        for node in self.tensor_network.train_nodes:
            self.tensor_params.append(nn.Parameter(node.tensor))
            node.tensor = self.tensor_params[-1]
        self.parametrized = True
    
    def state_dict(self, *args, **kwargs):
        """Returns the state dictionary of the layer."""
        if not self.parametrized:
            self.parametrize()
        return super(TensorNetworkLayer, self).state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Loads the state dictionary into the layer."""
        if not self.parametrized:
            self.parametrize()
        super(TensorNetworkLayer, self).load_state_dict(state_dict, *args, **kwargs)
        for node, param in zip(self.tensor_network.train_nodes, self.tensor_params):
            node.tensor = param
        self.parametrized = True
        return self

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
    
    def num_parameters(self):
        """Returns the number of parameters in the layer."""
        return sum(p.tensor.numel() for p in self.tensor_network.train_nodes)

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
    def __init__(self, num_carriages, bond_dim, num_patches, patch_pixels, output_shape, ring=False, convolution_bond=-1):
        """Initializes a TensorConvolutionTrainLayer."""
        if ring:
            raise NotImplementedError("Ring structure is not implemented for TensorConvolutionTrainLayer.")
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.num_patches = num_patches
        self.patch_pixels = patch_pixels
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring
        self.convolution_bond = convolution_bond

        self.output_labels = ('s',)

        # Create nodes
        x_nodes = []
        conv_blocks = []
        train_blocks = []
        for i in range(1, num_carriages+1):
            x_node = TensorNode((1, num_patches, patch_pixels), ['s', 'patches', 'patch_pixels'], name=f"X{i}")
            if convolution_bond > 0:
                conv_block = TensorNode((convolution_bond if i != 1 else 1, patch_pixels, convolution_bond if i != num_carriages else 1), [f'CB{i}', 'patch_pixels', f'CB{i+1}'], l=f'CB{i}', r=f'CB{i+1}', name=f"C{i}")
            else:
                conv_block = TensorNode((patch_pixels,), ['patch_pixels'], name=f"C{i}")
            train_block = TensorNode((bond_dim if i != 1 else 1, self.output_shape[i-1] if i <= len(self.output_shape) else 1, num_patches, bond_dim if i != num_carriages else 1), [f'r{i}', f'c{i}', 'patches', f'r{i+1}'], l=f'r{i}', r=f'r{i+1}', name=f"A{i}")
            x_nodes.append(x_node)
            conv_blocks.append(conv_block)
            train_blocks.append(train_block)
            if i < len(self.output_shape)+1:
                self.output_labels = self.output_labels + (f'c{i}',)
        
        self.nodes = []
        for i, (xn, cb, tb) in enumerate(zip(x_nodes, conv_blocks, train_blocks)):
            xn.connect(tb, 'patches')
            cb.connect(xn, 'patch_pixels')
            self.nodes.append(cb)
            self.nodes.append(tb)
        
        for i in range(1, num_carriages):
            train_blocks[i-1].connect(train_blocks[i], f'r{i+1}')

        # Connect convolution blocks horizontally if convolution_bond > 0
        if convolution_bond > 0:
            for i in range(1, num_carriages):
                conv_blocks[i-1].connect(conv_blocks[i], f'CB{i+1}')
        

        for n in train_blocks:
            n.squeeze()

        for n in conv_blocks:
            n.squeeze()
        
        # Create a TensorNetwork
        self.x_nodes = x_nodes
        self.conv_blocks = conv_blocks
        self.train_blocks = train_blocks
        self.labels = self.output_labels
        tensor_network = TensorNetwork(x_nodes, train_blocks, self.nodes, output_labels=self.labels)
        super(TensorConvolutionTrainLayer, self).__init__(tensor_network)
    
    def grow_cart(self, new_bond=None, new_convolution_bond=None):
        x_node_new = TensorNode(
            (1, self.num_patches, self.patch_pixels),
            ['s', 'patches', 'patch_pixels'],
            name=f"X{self.num_carriages+1}"
        )
        
        if new_bond is None:
            new_bond = self.bond_dim

        if new_convolution_bond is None:
            new_convolution_bond = self.convolution_bond

        train_tensor_new = torch.zeros((new_bond, 1, self.num_patches, 1))
        train_tensor_new[:, :, -1] = 1/new_bond
        train_block_new = TensorNode(
            train_tensor_new,
            [f'r{self.num_carriages+1}', f'c{self.num_carriages+1}', 'patches', f'r{self.num_carriages+2}'],
            l=f'r{self.num_carriages+1}', r=f'r{self.num_carriages+2}', name=f"A{self.num_carriages+1}"
        )

        # Connect new train block to x_node
        x_node_new.connect(train_block_new, 'patches')

        if new_convolution_bond > 0:
            conv_block_new = TensorNode(
                (new_convolution_bond if self.num_carriages != 1 else 1, self.patch_pixels, new_convolution_bond if self.num_carriages != self.num_carriages else 1),
                [f'CB{self.num_carriages+1}', 'patch_pixels', f'CB{self.num_carriages+2}'],
                l=f'CB{self.num_carriages+1}', r=f'CB{self.num_carriages+2}', name=f"C{self.num_carriages+1}"
            )
        else:
            conv_block_new = TensorNode(
                (self.patch_pixels,),
                ['patch_pixels'],
                name=f"C{self.num_carriages+1}"
            )

        # Connect new conv block to x_node
        x_node_new.connect(conv_block_new, 'patch_pixels')
        self.x_nodes.append(x_node_new)

        # Expand last node
        self.train_blocks[-1].expand_labels(self.train_blocks[-1].dim_labels + [f'r{self.num_carriages+1}'], self.train_blocks[-1].shape + (new_bond,))
        train_block_new.connect(self.train_blocks[-1], f'r{self.num_carriages+1}')
        train_block_new.squeeze()
        self.train_blocks.append(train_block_new)
        
        if new_convolution_bond > 0:
            self.conv_blocks[-1].expand_labels(self.conv_blocks[-1].dim_labels + [f'CB{self.num_carriages+1}'], self.conv_blocks[-1].shape + (new_convolution_bond,))
            self.conv_blocks[-1].connect(conv_block_new, f'CB{self.num_carriages+1}')
        conv_block_new.squeeze()
        self.conv_blocks.append(conv_block_new)
        
        self.num_carriages += 1

        self.tensor_network = TensorNetwork(self.x_nodes, self.train_blocks, self.tensor_network.train_nodes + [conv_block_new, train_block_new], output_labels=self.labels)

class TensorConvolutionGridTrainLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, num_layers, bond_dim, layer_bond, num_patches, patch_pixels, output_shape, ring=False, convolution_bond=-1):
        """Initializes a TensorConvolutionGridTrainLayer."""
        if ring:
            raise NotImplementedError("Ring structure is not implemented for TensorConvolutionGridTrainLayer.")
        self.num_carriages = num_carriages
        self.num_layers = num_layers
        self.bond_dim = bond_dim
        self.layer_bond = layer_bond
        self.num_patches = num_patches
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring
        self.convolution_bond = convolution_bond

        self.output_labels = ('s',)

        # Create input nodes (x_nodes) and convolution blocks (conv_blocks)
        x_nodes = []
        conv_blocks = []
        for i in range(1, num_carriages+1):
            x_node = TensorNode((1, num_patches, patch_pixels), ['s', 'patches', 'patch_pixels'], name=f"X{i}")
            if convolution_bond > 0:
                conv_block = TensorNode(
                    (convolution_bond if i != 1 else 1, patch_pixels, convolution_bond if i != num_carriages else 1),
                    [f'CB{i}', 'patch_pixels', f'CB{i+1}'],
                    l=f'CB{i}', r=f'CB{i+1}', name=f"C{i}"
                )
            else:
                conv_block = TensorNode((patch_pixels,), ['patch_pixels'], name=f"C{i}")
            x_nodes.append(x_node)
            conv_blocks.append(conv_block)

        # Create grid of train_blocks: shape [num_layers][num_carriages]
        train_blocks = []
        for l in range(num_layers):
            layer_blocks = []
            for i in range(1, num_carriages+1):
                # Only top layer gets output dims (c) and others get c_dim=1
                if l == num_layers - 1:
                    c_dim = self.output_shape[i-1] if i <= len(self.output_shape) else 1
                    c_label = f'c{i}' if i <= len(self.output_shape) else 'c'
                else:
                    c_dim = 1
                    c_label = 'c'
                left_bond = bond_dim if i != 1 else 1
                right_bond = bond_dim if i != num_carriages else 1
                if l == 0:
                    up_bond = layer_bond if num_layers > 1 else 1
                    labels = [f'v{l}_{i}', f'r{l}_{i}', c_label, 'patches', f'r{l}_{i+1}', f'v{l+1}_{i}']
                    shape = (
                        1,
                        left_bond,
                        c_dim,
                        num_patches,
                        right_bond,
                        up_bond
                    )
                else:
                    up_bond = layer_bond if l < num_layers-1 else 1
                    down_bond = layer_bond
                    labels = [f'v{l}_{i}', f'r{l}_{i}', c_label, f'r{l}_{i+1}', f'v{l+1}_{i}']
                    shape = (
                        down_bond,
                        left_bond,
                        c_dim,
                        right_bond,
                        up_bond
                    )
                l_label = f'r{l}_{i}'
                r_label = f'r{l}_{i+1}'
                block = TensorNode(shape, labels, l=l_label, r=r_label, name=f"A{l}_{i}")
                layer_blocks.append(block)
            train_blocks.append(layer_blocks)

        # Connect horizontally within each layer
        for l in range(num_layers):
            for i in range(1, num_carriages):
                train_blocks[l][i-1].connect(train_blocks[l][i], f'r{l}_{i+1}', priority=1)

        # Connect vertically between layers
        for l in range(num_layers-1):
            for i in range(num_carriages):
                train_blocks[l][i].connect(train_blocks[l+1][i], f'v{l+1}_{i+1}', priority=10)

        # Connect bottom layer train_blocks to x_nodes and conv_blocks
        for i in range(num_carriages):
            x_nodes[i].connect(train_blocks[0][i], 'patches')
            conv_blocks[i].connect(x_nodes[i], 'patch_pixels')

        # Connect convolution blocks horizontally if convolution_bond > 0
        if convolution_bond > 0:
            for i in range(1, num_carriages):
                conv_blocks[i-1].connect(conv_blocks[i], f'CB{i+1}')

        # Squeeze singleton dims
        for l in range(num_layers):
            for block in train_blocks[l]:
                block.squeeze()
        for cb in conv_blocks:
            cb.squeeze()

        # Set output labels for top layer train_blocks, using c{i} for each output dim
        self.output_labels = ('s',)
        for i in range(1, num_carriages+1):
            if (num_layers > 0) and (i <= len(self.output_shape)):
                self.output_labels = self.output_labels + (f'c{i}',)

        # Order nodes: for each column, bottom to top (conv_block, then train_blocks)
        self.nodes = []
        for i in range(num_carriages):
            self.nodes.append(conv_blocks[i])
            for l in range(num_layers):
                self.nodes.append(train_blocks[l][i])

        # Save for reference
        self.x_nodes = x_nodes
        self.conv_blocks = conv_blocks
        self.train_blocks = train_blocks
        self.labels = self.output_labels

        # Main nodes are the top layer train_blocks
        main_nodes = [train_blocks[-1][i] for i in range(num_carriages)]
        # Create a TensorNetwork
        tensor_network = TensorNetwork(x_nodes, main_nodes, self.nodes, output_labels=self.labels)
        super(TensorConvolutionGridTrainLayer, self).__init__(tensor_network)

from tensor.node import CPDTensorNode

class CPD(TensorNetworkLayer):
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
            if left == 1 or right == 1:
                block_tensor = (left, up, down, right)
            else:
                block_tensor = torch.zeros((left, up, down, right))
                for u in range(up):
                    for k in range(down):
                        block_tensor[:, u, k, :] = torch.diag(torch.randn(right))
            node = CPDTensorNode(block_tensor, [left_label, up_label, 'p', right_label], l=left_label, r=right_label, name=f"A{i}")
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
        super(CPD, self).__init__(tensor_network)