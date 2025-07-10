import torch
import torch.nn as nn
import numpy as np
from tensor.network import TensorNetwork, CPDNetwork
from tensor.node import TensorNode
from collections import defaultdict
from tensor.data_compression import train_concat

class MainNodeLayer(nn.Module):
    def __init__(self, N, r, f, output_shape=tuple(), down_label='p', constrict_bond=True, perturb=False, dtype=None):
        """Creates the main nodes for the layer."""
        super(MainNodeLayer, self).__init__()
        output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        labels = ['s']
        nodes = []

        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f) if constrict_bond else R
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f) if constrict_bond else R
            if left != 0:
                mx = left
            return (mx, b1)

        def build_perturb(rl, f, rr):
            if rl==rr:
                block = torch.diag_embed(torch.ones(rr, dtype=dtype)).unsqueeze(1)
            else:
                block = torch.ones(rl, rr, dtype=dtype).unsqueeze(1)

            blockf = torch.cat((torch.zeros(rl, f-1, rr), block), dim=1)
            return blockf.unsqueeze(1)

        if perturb:
            b0 = 0.02 * torch.randn((1, output_shape[0], f, r), dtype=dtype)
            bn = build_perturb(r, f, 1)
            left_stack = [b0]
            right_stack = [bn]
            middle = [b0, bn]
            for i in range(N-2):
                
                rl = left_stack[-1].shape[-1]
                rr = right_stack[0].shape[0]
                if i == N-3:
                    middle_block = build_perturb(rl, f, rr)
                    middle = [*left_stack, middle_block, *right_stack]
                left_stack.append(build_perturb(rl, f, r))
        else:
            b0 = build_left(1, f, r)
            bn = build_right(r, f, 1)
            left_stack = [b0]
            right_stack = [bn]
            middle = [b0, bn]
            for i in range(N-2):
                left_r = left_stack[-1][1]
                right_r = right_stack[0][0]
                if i == N-3:
                    middle_block = (left_r, right_r)
                    middle = [*left_stack, middle_block, *right_stack]
                if i % 2 == 0:
                    left_stack.append(build_left(left_r, f, r))
                else:
                    right_stack.insert(0, build_right(r, f, right_r))

        for i in range(1, N+1):
            if i-1 < len(output_shape):
                up = output_shape[i-1]
                up_label = f'c{i}'
                labels.append(up_label)
            else:
                up = 1
                up_label = 'c'
            down = f
            left_label = f'r{i}'
            right_label = f'r{i+1}'
            
            node_input = middle[i-1]
            if not perturb:
                left, right = node_input
                node_input = (left, up, down, right)

            node = TensorNode(node_input, [left_label, up_label, down_label.format(i), right_label], l=left_label, r=right_label, name=f"A{i}", dtype=dtype)
            nodes.append(node)
        
        self.nodes = nodes
        self.labels = labels

class NodeLayer(nn.Module):
    def __init__(self, N, size, labels, name='L{0}', dtype=None):
        """Creates the linear nodes for the layer."""
        super(NodeLayer, self).__init__()
        nodes = []
        for i in range(1, N+1):
            node = TensorNode(size, [l.format(i) for l in labels], name=name.format(i), dtype=dtype)
            nodes.append(node)
        self.nodes = nodes

class InputNodeLayer(NodeLayer):
    def __init__(self, N, f, label='p', dtype=None):
        """Creates the input nodes for the layer."""
        super(InputNodeLayer, self).__init__(N, (1, f), ['s', label], name='X{0}', dtype=dtype)

class TensorNetworkLayer(nn.Module):
    def __init__(self, tensor_network: TensorNetwork = None):
        """Initializes a TensorNetworkLayer."""
        super(TensorNetworkLayer, self).__init__()
        self.set_tensor_network(tensor_network)

    def set_tensor_network(self, tensor_network: TensorNetwork = None):
        """Sets the tensor network and labels for the layer."""
        self.tensor_network = tensor_network
        self.labels = tensor_network.output_labels if tensor_network is not None else None
        self.parametrized = False
        self.nodes = tensor_network.train_nodes if tensor_network is not None else []

    def node_states(self, detach=True):
        """Returns the state dictionary of the nodes."""
        tensor_params = {}
        for i, node in enumerate(self.tensor_network.train_nodes):
            if detach:
                tensor_params[f"tensor_param_{i}"] = node.tensor.detach().clone()
            else:
                tensor_params[f"tensor_param_{i}"] = node.tensor
        return tensor_params

    def load_node_states(self, tensor_params, set_value=False):
        """Loads the state dictionary into the nodes."""
        for i, node in enumerate(self.tensor_network.train_nodes):
            if f"tensor_param_{i}" in tensor_params:
                if set_value:
                    node.tensor = tensor_params[f"tensor_param_{i}"]
                else:
                    node.tensor.data.copy_(tensor_params[f"tensor_param_{i}"].detach().clone())
            else:
                raise ValueError(f"Missing parameter: tensor_param_{i}")
        
        self.tensor_network.reset_stacks()

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
    
    def zip_connect(self, nodes1, nodes2, label='p'):
        """Connects two lists of nodes with a zip connection."""
        if len(nodes1) != len(nodes2):
            raise ValueError("The number of nodes in both lists must be the same.")
        for i, (n1, n2) in enumerate(zip(nodes1, nodes2), 1):
            n1.connect(n2, label.format(i))
    
    def horizontal_connect(self, nodes):
        """Connects nodes horizontally."""
        if len(nodes) < 2:
            return
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            if n1.right_labels and n2.left_labels and n1.right_labels[0] != n2.left_labels[0]:
                raise ValueError(f"Right label of the first node does not match left label of the second node. Nodes: {n1.name}, {n2.name}")
            n1.connect(n2, n1.right_labels[0], priority=1)            

class TensorTrainLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, input_features, output_shape=tuple(), squeeze=True, constrict_bond=True, perturb=False, dtype=None, seed=None):
        """Initializes a TensorTrainLayer."""
        super(TensorTrainLayer, self).__init__()
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.input_features = input_features
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.main_node_layer = MainNodeLayer(
            num_carriages, bond_dim, input_features, output_shape=output_shape,
            down_label='p{0}', constrict_bond=constrict_bond, perturb=perturb, dtype=dtype
        )
        self.horizontal_connect(self.main_node_layer.nodes)
        self.input_node_layer = InputNodeLayer(num_carriages, input_features, label='p{0}', dtype=dtype)
        self.zip_connect(self.input_node_layer.nodes, self.main_node_layer.nodes, label='p{0}')
        
        if squeeze:
            for node in self.main_node_layer.nodes:
                node.squeeze(self.main_node_layer.labels)
        
        # Create a TensorNetwork
        tensor_network = TensorNetwork(self.input_node_layer.nodes, self.main_node_layer.nodes, output_labels=self.main_node_layer.labels)
        self.set_tensor_network(tensor_network)

def tensor_network_update(module):
    if isinstance(module, TensorTrainNN):
        node = module.tensor_network.train_nodes[module._cur_block_idx]
        step_tensor = module.tensor_network.solve_system(node, module._A_cur, module._b_cur, method=module._method, eps=module._eps)
        with torch.no_grad():
            p = module._parameters[f'tensor_node_{module._cur_block_idx}']
            p.copy_(p + step_tensor)
        module._cur_block_idx += 1
        module._A_cur = None
        module._b_cur = None
        if module._cur_block_idx >= len(module.tensor_network.train_nodes):
            module._cur_block_idx = 0
            module._eps = max(module._eps * 0.7, 4e-4)
            module._lmb = min(1 - (1 - module._lmb) * 0.8, 0.95)
            print(f"Updated parameters for all nodes, new eps: {module._eps}, new lmb: {module._lmb}")

class TensorTrainNN(TensorTrainLayer):
    def __init__(self, input_features, output_shape, N=3, r=8, squeeze=True, constrict_bond=True, perturb=False, dtype=None, seed=None, natural_gradient=True):
        """Initializes a TensorTrainNN."""
        super(TensorTrainNN, self).__init__(
            num_carriages=N, bond_dim=r, input_features=input_features+1,
            output_shape=output_shape, squeeze=squeeze, constrict_bond=constrict_bond,
            perturb=perturb, dtype=dtype, seed=seed
        )
        self._parameters = nn.ParameterDict()
        for i, node in enumerate(self.tensor_network.train_nodes):
            self._parameters[f'tensor_node_{i}'] = nn.Parameter(node.tensor, requires_grad=not natural_gradient)
            node.tensor = self._parameters[f'tensor_node_{i}']
        self.node_state_dict = self.node_states(detach=False)

        self._natural_gradient = natural_gradient
        self._cur_block_idx = 0
        self._method = "ridge_cholesky"
        self._eps = 1e-2
        self._lmb = 0.9
        self._A_cur = None
        self._b_cur = None
    
    def accumulate_gradient(self, node, d_loss, sqd_loss, lmb=0.9):
        A, b_vec = self.tensor_network.get_A_b(node, d_loss, sqd_loss)
        if self._A_cur is None or self._b_cur is None:
            self._A_cur = A
            self._b_cur = b_vec
        else:
            self._A_cur = lmb * self._A_cur + (1 - lmb) * A
            self._b_cur = lmb * self._b_cur + (1 - lmb) * b_vec
    
    def forward(self, x):
        self.load_node_states(self.node_state_dict, set_value=True)

        # Pad x with ones for the bias term
        x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

        out = self.tensor_network.forward(x)
        if self.labels is not None:
            out.permute_first(*self.labels)
        out = out.tensor

        if not self._natural_gradient:
            return out

        out = out.requires_grad_(True)
        hook_handle = None
        def _hook(d_loss):
            nonlocal hook_handle
            hook_handle.remove()
            B = []
            for i in range(d_loss.shape[-1]):
                g2 = torch.autograd.grad(
                    outputs=d_loss[..., i].sum(),
                    inputs=out,
                    retain_graph=True,
                    create_graph=False
                )[0]
                B.append(g2.unsqueeze(-2))
            sqd_loss = torch.cat(B, dim=-2)

            node = self.tensor_network.train_nodes[self._cur_block_idx]
            self.accumulate_gradient(node, d_loss, sqd_loss, lmb=self._lmb)

            return d_loss

        hook_handle = out.register_hook(_hook)
        return out

class TensorTrainLinearLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, input_features, linear_dim, output_shape=tuple(), squeeze=True, constrict_bond=True, perturb=False, dtype=None, seed=None):
        """Initializes a TensorTrainLinearLayer."""
        super(TensorTrainLinearLayer, self).__init__()
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.input_features = input_features
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.linear_dim = linear_dim

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.main_node_layer = MainNodeLayer(
            num_carriages, bond_dim, linear_dim, output_shape=output_shape,
            down_label='lin{0}', constrict_bond=constrict_bond, perturb=perturb, dtype=dtype
        )
        self.horizontal_connect(self.main_node_layer.nodes)
        self.linear_layer = NodeLayer(
            num_carriages, (linear_dim, input_features), labels=('lin{0}', 'p{0}'), dtype=dtype
        )
        self.zip_connect(self.linear_layer.nodes, self.main_node_layer.nodes, label='lin{0}')
        self.input_node_layer = InputNodeLayer(num_carriages, input_features, label='p{0}', dtype=dtype)
        self.zip_connect(self.input_node_layer.nodes, self.linear_layer.nodes, label='p{0}')
        if squeeze:
            for node in self.main_node_layer.nodes:
                node.squeeze(self.main_node_layer.labels)
        # Create a TensorNetwork (interleaving nodes, such that)
        tensor_network = TensorNetwork(
            self.input_node_layer.nodes, [n for column in zip(self.main_node_layer.nodes, self.linear_layer.nodes) for n in column],
            output_labels=self.main_node_layer.labels
        )
        self.set_tensor_network(tensor_network)

def concatenate_trains(tensor_layers):
    nodes_to_concat = defaultdict(list)
    for i, layer in enumerate(tensor_layers):
        for j, n in enumerate(layer.nodes):
            tensor_block = n.tensor
            if j == 0:
                tensor_block = tensor_block.unsqueeze(0)
            elif j == len(layer.nodes) - 1:
                tensor_block = tensor_block.unsqueeze(-1)
            if j >= len(layer.labels) - 1:
                tensor_block = tensor_block.unsqueeze(1)
            nodes_to_concat[i].append(tensor_block)
    
    
    train = nodes_to_concat[0]
    for i in range(1, len(tensor_layers)):
        train = train_concat(train, nodes_to_concat[i])

    train[0] = train[0] / len(tensor_layers)
    
    return TensorTrainLayer(num_carriages=len(train), bond_dim=tensor_layers[0].bond_dim, input_features=tensor_layers[0].input_features, output_shape=tensor_layers[0].output_shape, nodes=train, squeeze=True)

class TensorTrainDMRGInfiLayer(TensorNetworkLayer):
    def __init__(self, bond_dim, input_features, output_shape=tuple(), ring=False, squeeze=True, constrict_bond=True):
        """Initializes a TensorTrainLayer."""
        self.num_carriages = 2
        self.bond_dim = bond_dim
        self.input_features = input_features
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring

        # Create input nodes
        self.x_nodes = []
        x_node1 = TensorNode((1, input_features), ['s', 'pL1'], name=f"XL1")
        self.x_nodes.append(x_node1)
        x_node2 = TensorNode((1, input_features), ['s', 'pR1'], name=f"XR1")
        self.x_nodes.append(x_node2)

        # Create main nodes
        self.nodes = []
        self.labels = ['s']
        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f) if constrict_bond else R
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f) if constrict_bond else R
            if left != 0:
                mx = left
            return (mx, b1)

        b0 = build_left(1, input_features, bond_dim)
        bn = build_right(bond_dim, input_features, 1)
        left_stack = [b0]
        right_stack = [bn]
        middle = [b0, bn]
        for i in range(self.num_carriages-2):
            b0 = left_stack[-1][1]
            b1 = right_stack[0][0]
            if i == self.num_carriages-3:
                middle_block = (b0, b1)
                middle = [*left_stack, middle_block, *right_stack]
            if i % 2 == 0:
                left_stack.append(build_left(b0, input_features, bond_dim))
            else:
                right_stack.insert(0, build_right(bond_dim, input_features, b1))
        self.ranks = middle

        up = self.output_shape[0]
        up_label = f'c1'
        self.labels.append(up_label)
        down = input_features

        left, right = self.ranks[0]

        node1 = TensorNode((up, down, right), [up_label, 'pL1', 'r1'], r='r1', name=f"AL1")
        node1.connect(x_node1, 'pL1', priority=2)
        self.nodes.append(node1)

        down = input_features
        left, right = self.ranks[1]

        node2 = TensorNode((left, down), ['r1', 'pR1'], l='r1', name=f"AR1")
        node2.connect(x_node2, 'pR1', priority=2)
        self.nodes.append(node2)

        node1.connect(node2, 'r1', priority=0)

        # Squeeze singleton dimensions
        if squeeze:
            for node in self.nodes:
                node.squeeze(self.labels)
        # Create a TensorNetwork
        tensor_network = TensorNetwork(self.x_nodes, self.nodes, output_labels=self.labels)
        super(TensorTrainDMRGInfiLayer, self).__init__(tensor_network)

    def grow_middle(self):
        phys_left = f'pL{self.num_carriages}'
        phys_right = f'pR{self.num_carriages}'
        x_node_new1 = TensorNode(
            (1, self.input_features),
            ['s', phys_left],
            name=f"XL{self.num_carriages}"
        )

        x_node_new2 = TensorNode(
            (1, self.input_features),
            ['s', phys_right],
            name=f"XR{self.num_carriages}"
        )

        middle_left = self.nodes[self.num_carriages//2-1]
        middle_right = self.nodes[self.num_carriages//2]


        left_label_name = middle_left.right_labels[0]
        for con in list(middle_left.connections.keys()):
            if left_label_name == con:
                del middle_left.connections[con]
        left_label_name += 'L'
        middle_left.right_labels = [left_label_name]
        middle_left.dim_labels[-1] = left_label_name

        right_label_name = middle_right.left_labels[0]
        for con in list(middle_right.connections.keys()):
            if right_label_name == con:
                del middle_right.connections[con]
        right_label_name += 'R'
        middle_right.left_labels = [right_label_name]
        middle_right.dim_labels[0] = right_label_name

        new_bond1 = middle_left.dim_size(left_label_name)
        new_bond2 = middle_right.dim_size(right_label_name)

        train_block_new = TensorNode(
            (new_bond1, 1, self.input_features, self.input_features, new_bond2),
            [left_label_name, f'c{self.num_carriages}', phys_left, phys_right, right_label_name],
            l=left_label_name, r=right_label_name, name=f"D{self.num_carriages}"
        )

        # Connect new train block to x_node
        x_node_new1.connect(train_block_new, phys_left)
        x_node_new2.connect(train_block_new, phys_right)
        self.x_nodes.insert(self.num_carriages//2, x_node_new2)
        self.x_nodes.insert(self.num_carriages//2, x_node_new1)

        train_block_new.connect(middle_left, left_label_name)
        train_block_new.connect(middle_right, right_label_name)
        train_block_new.squeeze()
        # Insert in the middle
        self.nodes.insert(self.num_carriages//2, train_block_new)

        self.num_carriages += 1
        self.tensor_network = TensorNetwork(self.x_nodes, self.nodes, train_nodes=[train_block_new], output_labels=self.labels)
        self.to(middle_left.tensor.device)

    def split_node(self, left_labels, right_labels, rank, err=None, is_last=False):
        node = self.nodes[self.num_carriages//2]
        cur_left_label = node.left_labels[0]
        cur_right_label = node.right_labels[0]
        # Permute such that left_labels are first
        node.permute_first(*left_labels)
        node.permute_last(*right_labels)
        # Flatten as a matrix
        matrix = node.tensor.reshape(np.prod([node.dim_size(l) for l in left_labels]), np.prod([node.dim_size(l) for l in right_labels]))
        # SVD of the matrix
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)
        if is_last:
            v = s.diag() @ v

        # Reshape u and v to tensors
        u = u.reshape([node.dim_size(l) for l in left_labels] + [u.shape[1]])
        v = v.reshape([v.shape[0]] + [node.dim_size(l) for l in right_labels])


        # Truncate u and v to the given error
        s_cumsum = torch.flip(s,dims=[0]).cumsum(0)
        if err is not None:
            rank = max(min(rank, (s_cumsum > err).sum()), 1)
        split_err = s_cumsum[-rank]
        u = u[..., :rank]
        v = v[:rank]

        # Create new nodes
        new_node1 = TensorNode(u, left_labels + [f'r{self.num_carriages}'], r=f'r{self.num_carriages}', l=cur_left_label, name=f"AL{self.num_carriages}")
        new_node2 = TensorNode(v, [f'r{self.num_carriages}'] + right_labels, r=cur_right_label, l=f'r{self.num_carriages}', name=f"AR{self.num_carriages}")

        # Connect the new nodes to the old node neighbors
        for l in node.left_labels:
            if l in node.connections:
                node.connections[l].connect(new_node1, l)
        for l in node.right_labels:
            if l in node.connections:
                node.connections[l].connect(new_node2, l)

        # Remove connections with the old node
        for con in list(node.connections.keys()):
            if con in left_labels or con in right_labels:
                del node.connections[con]

        # Connect new nodes in middle
        new_node1.connect(new_node2, f'r{self.num_carriages}')
        # Connect new nodes to x_nodes
        x_node1 = self.x_nodes[self.num_carriages//2]
        x_node2 = self.x_nodes[self.num_carriages//2+1]
        # Remove all connections from x_nodes
        x_node1.reset_connections()
        x_node2.reset_connections()
        x_node1.connect(new_node1, x_node1.dim_labels[1])
        x_node2.connect(new_node2, x_node2.dim_labels[1])
        # Add new nodes to the list
        self.nodes.insert(self.num_carriages//2, new_node2)
        self.nodes.insert(self.num_carriages//2, new_node1)
        # Update the number of carriages
        self.num_carriages += 1
        # Remove the old node
        self.nodes.remove(node)
        # Update the tensor network
        self.tensor_network = TensorNetwork(self.x_nodes, self.nodes, train_nodes=[], output_labels=self.labels)
        self.to(node.tensor.device)
        return split_err


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
        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f)
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f)
            if left != 0:
                mx = left
            return (mx, b1)

        b0 = build_left(1, input_features, bond_dim)
        bn = build_right(bond_dim, input_features, 1)
        left_stack = [b0]
        right_stack = [bn]
        middle = [b0, bn]
        for i in range(num_carriages-2):
            b0 = left_stack[-1][1]
            b1 = right_stack[0][0]
            if i == num_carriages-3:
                middle_block = (b0, b1)
                middle = [*left_stack, middle_block, *right_stack]
            if i % 2 == 0:
                left_stack.append(build_left(b0, input_features, bond_dim))
            else:
                right_stack.insert(0, build_right(bond_dim, input_features, b1))
        self.ranks = middle
        for i in range(num_carriages):
            left_label = 'rr' if ring and i == 0 else f'r{i}'
            right_label = 'rr' if ring and i == num_carriages - 1 else f'r{i+1}'

            left_dim, right_dim = self.ranks[i]

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
    def __init__(self, num_carriages, bond_dim, num_patches, patch_pixels, output_shape, ring=False, convolution_bond=-1, dtype=None, constrict_bond=True, perturb=False):
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
        if perturb:
            def build_perturb(rl, f, rr):
                if rl==rr:
                    block = torch.diag_embed(torch.ones(rr, dtype=dtype)).unsqueeze(1)
                else:
                    block = torch.ones(rl, rr, dtype=dtype).unsqueeze(1)

                blockf = torch.cat((torch.zeros(rl, f-1, rr), block), dim=1)
                return blockf
            
            b0 = torch.randn((1, num_patches, bond_dim), dtype=dtype) #build_perturb(1, num_patches, bond_dim)
            bn = build_perturb(bond_dim, num_patches, 1)
            left_stack = [b0]
            right_stack = [bn]
            middle = [b0, bn]
            for i in range(num_carriages-2):
                b0 = left_stack[-1].shape[-1]
                b1 = right_stack[0].shape[0]
                if i == num_carriages-3:
                    middle_block = build_perturb(b0, num_patches, b1)
                    middle = [*left_stack, middle_block, *right_stack]
                left_stack.append(build_perturb(b0, num_patches, bond_dim))

            blocks = [b.unsqueeze(1) for b in middle]
        else:
            blocks = []
            for i in range(1, num_carriages+1):
                blocks.append((bond_dim if i != 1 else 1, self.output_shape[i-1] if i <= len(self.output_shape) else 1, num_patches, bond_dim if i != num_carriages else 1))

        for i in range(1, num_carriages+1):
            if i-1 < len(self.output_shape):
                up_label = f'c{i}'
            else:
                up_label = 'c'
            left_label = f'r{i}'
            right_label = f'r{i+1}'

            x_node = TensorNode((1, num_patches, patch_pixels), ['s', 'patches', 'patch_pixels'], name=f"X{i}")
            if convolution_bond > 0:
                conv_block = TensorNode((convolution_bond if i != 1 else 1, patch_pixels, convolution_bond if i != num_carriages else 1), [f'CB{i}', 'patch_pixels', f'CB{i+1}'], l=f'CB{i}', r=f'CB{i+1}', name=f"C{i}")
            else:
                conv_block = TensorNode((patch_pixels,), ['patch_pixels'], name=f"C{i}")
            train_block = TensorNode(blocks[i-1], [left_label, up_label, 'patches', right_label], l=f'r{i}', r=f'r{i+1}', name=f"A{i}")
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
    def __init__(self, num_carriages, num_layers, bond_dim, lin_dim, lin_bond, num_patches, patch_pixels, output_shape, ring=False, convolution_bond=-1):
        """Initializes a TensorConvolutionGridTrainLayer."""
        if ring:
            raise NotImplementedError("Ring structure is not implemented for TensorConvolutionGridTrainLayer.")
        self.num_carriages = num_carriages
        self.num_layers = num_layers
        self.bond_dim = bond_dim
        self.lin_dim = lin_dim
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
                left_bond = max(1, (bond_dim if l == num_layers - 1 else lin_bond) if i != 1 else 1)
                right_bond = max(1, (bond_dim if l == num_layers - 1 else lin_bond) if i != num_carriages else 1)
                if l == 0:
                    up_bond = lin_dim if num_layers > 1 else 1
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
                    up_bond = lin_dim if l < num_layers-1 else 1
                    down_bond = lin_dim
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
            if lin_bond <= 0 and l != num_layers - 1:
                continue
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

class TensorTrainLinearLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, input_features, linear_dim, linear_bond=1,
                 output_shape=tuple(), ring=False, squeeze=True, connect_linear=False):
        """Initializes a TensorTrainLinearLayer with an intermediate linear block."""
        self.num_carriages = num_carriages
        self.bond_dim = bond_dim
        self.input_features = input_features
        self.linear_dim = linear_dim
        self.linear_bond = linear_bond
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.ring = ring
        self.connect_linear = connect_linear

        # create input nodes
        self.x_nodes = []
        for i in range(1, num_carriages+1):
            x = TensorNode((1, input_features), ['s', 'p'], name=f"X{i}")
            self.x_nodes.append(x)

        # build linear and train blocks
        self.linear_blocks = []
        self.train_blocks = []
        self.nodes = []  # main nodes are train blocks
        self.labels = ['s']
        self.train_nodes = []
        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f)
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f)
            if left != 0:
                mx = left
            return (mx, b1)

        b0 = build_left(1, input_features, bond_dim)
        bn = build_right(bond_dim, input_features, 1)
        left_stack = [b0]
        right_stack = [bn]
        middle = [b0, bn]
        for i in range(num_carriages-2):
            b0 = left_stack[-1][1]
            b1 = right_stack[0][0]
            if i == num_carriages-3:
                middle_block = (b0, b1)
                middle = [*left_stack, middle_block, *right_stack]
            if i % 2 == 0:
                left_stack.append(build_left(b0, input_features, bond_dim))
            else:
                right_stack.insert(0, build_right(bond_dim, input_features, b1))
        self.ranks = middle
        for i in range(1, num_carriages+1):
            # linear block shape/labels
            if self.connect_linear:
                shape_lin = (linear_bond if i != 1 else 1, input_features, linear_dim, linear_bond if i != num_carriages else 1)
                labels_lin = [f'l{i}', 'p', 'm', f'l{i+1}']
                lin = TensorNode(shape_lin, labels_lin, l=f'l{i}', r=f'l{i+1}', name=f"L{i}")
            else:
                shape_lin = (input_features, linear_dim)
                labels_lin = ['p', 'm']
                lin = TensorNode(shape_lin, labels_lin, name=f"L{i}")
            self.train_nodes.append(lin)

            # Connect x_node to linear block explicitly
            self.x_nodes[i-1].connect(lin, 'p', priority=2)

            # optional connect between linear blocks
            if self.connect_linear and i>1:
                self.linear_blocks[-1].connect(lin, f'l{i}', priority=1)
            self.linear_blocks.append(lin)

            # prepare train block
            # output dims
            if i-1 < len(self.output_shape):
                up = self.output_shape[i-1]
                up_label = f'c{i}'
                self.labels.append(up_label)
            else:
                up = 1
                up_label = 'c'
            # bond dims
            left_label = 'rr' if ring and i==1 else f'r{i}'
            right_label = 'rr' if ring and i==num_carriages else f'r{i+1}'
            left, right = self.ranks[i-1]
            # create and connect train node
            tr = TensorNode((left, up, linear_dim, right),
                            [left_label, up_label, 'm', right_label],
                            l=left_label, r=right_label, name=f"A{i}")
            self.train_nodes.append(tr)

            # Connect linear block to train block explicitly
            tr.connect(lin, 'm', priority=2)

            if i>1:
                self.train_blocks[-1].connect(tr, left_label, priority=1)
            if ring and i==num_carriages:
                tr.connect(self.train_blocks[0], right_label, priority=0)
            self.train_blocks.append(tr)
            self.nodes.append(tr)

        # squeeze singleton dims
        if squeeze:
            for node in self.linear_blocks + self.nodes:
                node.squeeze(self.labels)

        # build tensor network including linear and train as train_nodes
        tensor_network = TensorNetwork(
            self.x_nodes, self.nodes,
            train_nodes=self.train_nodes,
            output_labels=self.labels
        )
        super(TensorTrainLinearLayer, self).__init__(tensor_network)

class TensorTrainSplitInputLayer(TensorNetworkLayer):
    def __init__(self, num_wagons, bond_dim, input_shape=tuple(), output_shape=tuple(), axle_bond=1):
        """Initializes a TensorTrainSplitInputLayer."""
        num_input_dims = len(input_shape)
        self.num_wagons = num_wagons
        self.bond_dim = bond_dim
        self.axle_bond = axle_bond
        self.input_shape = input_shape
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.labels = ['s']

        self.x_nodes = []
        self.nodes = []
        for i in range(num_wagons):
            for j in range(num_input_dims):
                idx = i * num_input_dims + j
                if j == 0:
                    x_node = TensorNode((1,)+input_shape, ['s']+[f'I{i * num_input_dims + l}' for l in range(num_input_dims)], name=f"X{i}")
                    self.x_nodes.append(x_node)
                if idx < len(self.output_shape):
                    up = self.output_shape[idx]
                    up_label = f'c{idx}'
                    self.labels.append(up_label)
                else:
                    up = 1
                    up_label = 'c'
                down = input_shape[j]
                down_label = f'I{idx}'

                left_label = f'r{idx}'
                right_label = f'r{idx+1}'
                # First cart in total
                if (i == 0 and j == 0):
                    left = 1
                    right = bond_dim
                # Last cart in total
                elif (i == num_wagons-1 and j == num_input_dims-1):
                    left = bond_dim
                    right = 1
                # First cart in wagon
                elif (j == 0):
                    left = axle_bond
                    right = bond_dim
                # Last cart in wagon
                elif (j == num_input_dims-1):
                    left = bond_dim
                    right = axle_bond
                # Middle cart in wagon
                else:
                    left = bond_dim
                    right = bond_dim
                # Create the node
                node = TensorNode((left, up, down, right), [left_label, up_label, down_label, right_label], l=left_label, r=right_label, name=f"A{idx}")
                # If not the first wagon, connect to the previous node
                if i > 0 or j > 0:
                    self.nodes[-1].connect(node, left_label, priority=1)
                node.connect(self.x_nodes[i], down_label, priority=2)
                self.nodes.append(node)

        # Squeeze singleton dimensions
        for node in self.nodes:
            node.squeeze(self.labels)
        # Create a TensorNetwork
        tensor_network = TensorNetwork(self.x_nodes, self.nodes, output_labels=self.labels)
        super(TensorTrainSplitInputLayer, self).__init__(tensor_network)

class ComplexTensorTrainLayer(TensorNetworkLayer):
    def __init__(self, num_carriages, bond_dim, input_features, output_shape=tuple(), ring=False, squeeze=True, constrict_bond=True):
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
        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f) if constrict_bond else R
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f) if constrict_bond else R
            if left != 0:
                mx = left
            return (mx, b1)

        b0 = build_left(1, input_features, bond_dim)
        bn = build_right(bond_dim, input_features, 1)
        left_stack = [b0]
        right_stack = [bn]
        middle = [b0, bn]
        for i in range(num_carriages-2):
            b0 = left_stack[-1][1]
            b1 = right_stack[0][0]
            if i == num_carriages-3:
                middle_block = (b0, b1)
                middle = [*left_stack, middle_block, *right_stack]
            if i % 2 == 0:
                left_stack.append(build_left(b0, input_features, bond_dim))
            else:
                right_stack.insert(0, build_right(bond_dim, input_features, b1))
        self.ranks = middle
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

            left, right = self.ranks[i-1]

            tensor_node = torch.randn((left, up, down, right), dtype=torch.complex128)
            node = TensorNode(tensor_node, [left_label, up_label, 'p', right_label], l=left_label, r=right_label, name=f"A{i}")
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
        super(ComplexTensorTrainLayer, self).__init__(tensor_network)

class TensorConvOperatorLayer(TensorNetworkLayer):
    def __init__(self,
                 operator,
                 input_features,
                 bond_dim,
                 num_carriages,
                 num_patches,
                 patch_pixels,
                 output_shape,
                 ring=False,
                 convolution_bond=-1):
        """
        A layer that first applies a conv block (C),
        then an input node (X), then your operator (O),
        then the train block (A).
        """
        # Store args
        self.operator       = operator
        self.input_features = input_features
        self.bond_dim       = bond_dim
        self.num_carriages  = num_carriages
        self.num_patches    = num_patches
        self.patch_pixels   = patch_pixels
        self.output_shape   = (output_shape
                               if isinstance(output_shape, tuple)
                               else (output_shape,))
        self.ring           = ring
        self.convolution_bond = convolution_bond

        # Final network’s output labels
        self.output_labels = ('s',)

        # 1) Build all the conv‐blocks C[i]
        self.conv_blocks = []
        for i in range(num_carriages):
            if convolution_bond > 0:
                C = TensorNode(
                    (convolution_bond if i != 0 else 1,
                     patch_pixels,
                     convolution_bond if i != num_carriages-1 else 1),
                    [f'CB{i+1}', 'patch_pixels', f'CB{i+2}'],
                    l=f'CB{i+1}', r=f'CB{i+2}',
                    name=f"C{i}")
            else:
                C = TensorNode(
                    (patch_pixels,),
                    ['patch_pixels'],
                    name=f"C{i}")
            self.conv_blocks.append(C)

        # 2) Build all the input nodes X[i]
        self.x_nodes = []
        for i in range(num_carriages):
            X = TensorNode(
                (1, num_patches, input_features),
                ['s', 'patches', f'din{i}'],
                name=f"X{i}")
            self.x_nodes.append(X)

        # 3) Build all the operator nodes O[i]
        self.op_nodes = []
        for i in range(num_carriages):
            # pick the i-th operator tensor
            if isinstance(operator, (list, tuple)):
                O_tens = operator[i]
            else:
                O_tens = operator
            O = TensorNode(
                O_tens,
                (f'din{i}', f'u{i}'),
                name=f"O{i}")
            self.op_nodes.append(O)

        # 4) Build all the train blocks A[i]
        self.train_blocks = []
        for i in range(num_carriages):
            out_dim = (self.output_shape[i]
                       if i < len(self.output_shape)
                       else 1)
            A = TensorNode(
                (bond_dim if i != 0 else 1,
                 out_dim,
                 num_patches,
                 bond_dim if i != num_carriages-1 else 1),
                [f'r{i+1}', f'c{i+1}', 'patches', f'r{i+2}'],
                l=f'r{i+1}', r=f'r{i+2}',
                name=f"A{i}")
            self.train_blocks.append(A)
            self.output_labels += (f'c{i+1}',)

        # 5) Wire them C → X → O → A
        for i in range(num_carriages):
            C = self.conv_blocks[i]
            X = self.x_nodes[i]
            O = self.op_nodes[i]
            A = self.train_blocks[i]

            C.connect(X, 'patch_pixels')            # C → X
            X.connect(O, f'din{i}')                 # X → O
            O.connect(A, f'u{i}')                   # O → A

        # 6) Chain the train blocks along their r-legs
        for i in range(num_carriages-1):
            self.train_blocks[i].connect(
                self.train_blocks[i+1],
                f'r{i+2}')

        # 7) Optionally close the conv‐block ring
        if self.convolution_bond > 0 and ring:
            for i in range(num_carriages-1):
                self.conv_blocks[i].connect(
                    self.conv_blocks[i+1],
                    f'CB{i+2}')
            self.conv_blocks[-1].connect(
                self.conv_blocks[0],
                'CB1')

        # 8) Squeeze out all singleton dims
        for n in (self.conv_blocks
                  + self.x_nodes
                  + self.op_nodes
                  + self.train_blocks):
            n.squeeze()

        # 9) Build and hand off the TensorNetwork
        network = TensorNetwork(
            self.x_nodes,
            self.train_blocks,
            output_labels=self.output_labels)
        super(TensorConvOperatorLayer, self).__init__(
            network,
            labels=self.output_labels)

class CompressedTensorTrainLayer(TensorNetworkLayer):
    def __init__(self, data_blocks, bond_dim, output_shape=tuple(), constrict_bond=True, perturb=False, seed=None):
        """Initializes a TensorTrainLayer."""
        N = len(data_blocks)
        self.num_carriages = N
        self.bond_dim = bond_dim
        self.input_features = data_blocks[0].shape[1]
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Create input nodes
        self.x_nodes = []
        self.physical_dims = []
        for i in range(1, N+1):
            x_node = TensorNode(data_blocks[i-1], [f'k{i}', f'p{i}', 'd', f'k{i+1}' if i < N else 's'], name=f"X{i}", l=f'k{i}', r=f'k{i+1}' if i < N else None)

            if self.x_nodes:
                x_node.connect(self.x_nodes[-1], f'k{i}', priority=1)
            self.x_nodes.append(x_node)
            self.physical_dims.append(data_blocks[i-1].shape[1])
        
        # Create main nodes
        self.nodes = []
        self.labels = ['s']

        def build_left(b0, f, R, right=0):
            mx = min(R, b0*f) if constrict_bond else R
            if right != 0:
                mx = right
            return (b0, mx)

        def build_right(R, f, b1, left=0):
            mx = min(R, b1*f) if constrict_bond else R
            if left != 0:
                mx = left
            return (mx, b1)

        def build_perturb(rl, f, rr):
            if rl==rr:
                block = torch.diag_embed(torch.ones(rr)).unsqueeze(1)
            else:
                block = torch.ones(rl, rr).unsqueeze(1)

            blockf = torch.cat((torch.zeros(rl, f-1, rr), block), dim=1)
            return blockf

        if perturb:
            b0 = build_perturb(1, self.physical_dims[0], bond_dim)#torch.randn((1, self.input_features, bond_dim)
            bn = build_perturb(bond_dim, self.physical_dims[-1], 1)
            left_stack = [b0]
            right_stack = [bn]
            middle = [b0, bn]
            for i in range(N-2):
                
                b0 = left_stack[-1].shape[-1]
                b1 = right_stack[0].shape[0]
                if i == N-3:
                    middle_block = build_perturb(b0, self.physical_dims[i+1], b1)
                    middle = [*left_stack, middle_block, *right_stack]
                left_stack.append(build_perturb(b0, self.physical_dims[i+1], bond_dim))

            self.pert_nodes = middle

            for i in range(1, N+1):
                if i-1 < len(self.output_shape):
                    up = self.output_shape[i-1]
                    up_label = f'c{i}'
                    self.labels.append(up_label)
                else:
                    up = 1
                    up_label = 'c'
                left_label = f'r{i}'
                right_label = f'r{i+1}'

                node = TensorNode(self.pert_nodes[i-1].unsqueeze(1), [left_label, up_label, f'p{i}', right_label], l=left_label, r=right_label, name=f"A{i}")
                if i > 1:
                    self.nodes[-1].connect(node, left_label, priority=1)
                node.connect(self.x_nodes[i-1], f'p{i}', priority=2)
                self.nodes.append(node)
        else:
            b0 = build_left(1, self.physical_dims[0], bond_dim)
            bn = build_right(bond_dim, self.physical_dims[-1], 1)
            left_stack = [b0]
            right_stack = [bn]
            middle = [b0, bn]
            for i in range(N-2):
                b0 = left_stack[-1][1]
                b1 = right_stack[0][0]
                if i == N-3:
                    middle_block = (b0, b1)
                    middle = [*left_stack, middle_block, *right_stack]
                if i % 2 == 0:
                    left_stack.append(build_left(b0, self.physical_dims[i+1], bond_dim))
                else:
                    right_stack.insert(0, build_right(bond_dim, self.physical_dims[i+1], b1))

            self.ranks = middle
            for i in range(1, N+1):
                if i-1 < len(self.output_shape):
                    up = self.output_shape[i-1]
                    up_label = f'c{i}'
                    self.labels.append(up_label)
                else:
                    up = 1
                    up_label = 'c'
                down = self.physical_dims[i-1]
                left_label = f'r{i}'
                right_label = f'r{i+1}'

                left, right = self.ranks[i-1]

                node = TensorNode((left, up, down, right), [left_label, up_label, f'p{i}', right_label], l=left_label, r=right_label, name=f"A{i}")
                if i > 1:
                    self.nodes[-1].connect(node, left_label, priority=1)
                node.connect(self.x_nodes[i-1], f'p{i}', priority=2)
                self.nodes.append(node)

        # Squeeze singleton dimensions
        for node in self.nodes:
            node.squeeze(self.labels)
        for x_node in self.x_nodes:
            x_node.squeeze(('s',))
        # Create a TensorNetwork
        tensor_network = TensorNetwork(self.x_nodes, self.nodes, output_labels=self.labels)
        super(CompressedTensorTrainLayer, self).__init__(tensor_network)

class CPDLayer(TensorNetworkLayer):
    def __init__(self, num_factors, rank, input_features, output_shape=tuple(), dtype=None):
        """
        Canonical Polyadic Decomposition (CPD) Layer.
        Args:
            num_factors: Number of factors (e.g., 3 for a 3-way tensor).
            rank: CP rank (number of components).
            input_features: Number of physical/input features per factor.
            output_shape: Tuple of output dimensions (optional, default empty).
            dtype: torch dtype for tensors (optional).
        """
        self.num_factors = num_factors
        self.rank = rank
        self.input_features = input_features
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)

        # 1. Create input nodes
        self.x_nodes = []
        for i in range(1, num_factors + 1):
            x_node = TensorNode(
                (1, input_features),
                ['SAMPLE', 'PHYS'],
                name=f"X{i}"
            )
            self.x_nodes.append(x_node)

        # 2. Create factor nodes
        self.nodes = []
        self.labels = ['SAMPLE']
        for i in range(1, num_factors + 1):
            # Output dimension for this factor (default 1 if not specified)
            out_dim = self.output_shape[i-1] if i-1 < len(self.output_shape) else 1
            # For the first factor, add an OUT leg for output, otherwise not
            if i == 1:
                node = TensorNode(
                    (rank, input_features, out_dim),
                    ['BOND', 'PHYS', 'OUT'],
                    name=f"A{i}"
                )
                self.labels.append('OUT')
            else:
                node = TensorNode(
                    (rank, input_features),
                    ['BOND', 'PHYS'],
                    name=f"A{i}"
                )
            self.nodes.append(node)

        # 3. Connect input nodes to factor nodes along 'PHYS'
        for x, a in zip(self.x_nodes, self.nodes):
            x.connect(a, 'PHYS')

        # 4. Build the TensorNetwork
        tensor_network = CPDNetwork(self.x_nodes, self.nodes, output_labels=tuple(self.labels), sample_dim='SAMPLE')
        super(CPDLayer, self).__init__(tensor_network)
