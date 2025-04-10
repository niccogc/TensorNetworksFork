import torch
import string
from collections import deque
from tensor_node import TensorNode
from tensor_utils import tensor_gradient_solver

class TensorNetwork:
    def __init__(self, input_nodes, main_nodes, output_labels=('s',), sample_dim='s'):
        """Initializes a TensorNetwork with ordered input and main nodes."""
        self.input_nodes = input_nodes
        self.main_nodes = main_nodes
        self.node_indices = {node: i for i, node in enumerate(main_nodes)}
        self.nodes = self._discover_nodes()
        self.left_stacks = None
        self.right_stacks = None
        self.output_labels = output_labels
        self.sample_dim = sample_dim

    def cuda(self):
        """Moves all tensors to the GPU."""
        for node in self.nodes:
            node.cuda()
        return self

    def to(self, device=None, dtype=None):
        """Moves all tensors to the given device and dtype."""
        for node in self.nodes:
            node.to(device=device, dtype=dtype)
        return self

    def _discover_nodes(self):
        """Uses BFS to find all connected nodes from input and main nodes."""
        discovered = set(self.input_nodes) | set(self.main_nodes)
        queue = deque(discovered)

        while queue:
            node = queue.popleft()
            for connected_node in node.connections.values():
                if connected_node not in discovered:
                    discovered.add(connected_node)
                    queue.append(connected_node)

        return list(sorted(discovered, key=lambda n: n.name))

    def compute_stack(self, direction="left", exclude_nodes=set()):
        """Computes left or right stacks iteratively."""
        stack_dict = {}
        nodes = self.main_nodes if direction == "left" else reversed(self.main_nodes)

        prev_stack = None

        for node in nodes:
            contracted = node.contract_vertically(exclude=exclude_nodes)
            if prev_stack is not None:
                contracted = prev_stack.contract_with(contracted, contracted.get_connecting_labels(prev_stack))
            stack_dict[node] = contracted
            prev_stack = contracted

        return stack_dict

    def recompute_all_stacks(self, exclude_nodes=set()):
        """Computes left and right stacks for each node."""
        self.left_stacks = self.compute_stack("left", exclude_nodes)
        self.right_stacks = self.compute_stack("right", exclude_nodes)

    def get_stacks(self, node):
        """Returns the left and right stacks for the given node."""
        index = self.node_indices[node]
        left_stack = self.left_stacks[self.main_nodes[index - 1]] if index > 0 else None
        right_stack = self.right_stacks[self.main_nodes[index + 1]] if index < len(self.main_nodes) - 1 else None
        return left_stack, right_stack

    def compute_jacobian_stack(self, node) -> TensorNode:
        """Computes the jacobian of a node by contracting all non-left/right connections separately."""
        contracted_nodes = [next_node.contract_vertically(exclude={node}) for label, next_node in node.connections.items() if label not in node.left_labels + node.right_labels]

        left_stack, right_stack = self.get_stacks(node)
        contracted_order = []
        if left_stack:
            contracted_order.append(left_stack)
        contracted_order.extend(contracted_nodes)
        if right_stack:
            contracted_order.append(right_stack)

        contracted = contracted_order[0]
        for i in range(1, len(contracted_order)):
            contracted = contracted.contract_with(contracted_order[i], contracted_order[i].get_connecting_labels(contracted))

        return contracted

    def forward(self, x):
        """Computes the forward pass of the network."""
        self.set_input(x)

        if self.left_stacks is None or self.right_stacks is None:
            self.recompute_all_stacks()
        node = self.main_nodes[0]
        contracted = node.contract_vertically()
        left_stack, right_stack = self.get_stacks(node)
        if left_stack:
            contracted = left_stack.contract_with(contracted, left_stack.right_labels)
        if right_stack:
            contracted = right_stack.contract_with(contracted, right_stack.left_labels)
        return contracted

    def node_forward(self, node):
        """Computes the forward pass of the network starting from the given node."""
        if self.left_stacks is None or self.right_stacks is None:
            self.recompute_all_stacks()
        contracted = node.contract_vertically()
        left_stack, right_stack = self.get_stacks(node)
        if left_stack:
            contracted = left_stack.contract_with(contracted, left_stack.right_labels)
        if right_stack:
            contracted = right_stack.contract_with(contracted, right_stack.left_labels)
        return contracted

    def left_update_stacks(self, node):
        """Updates the left stacks."""
        index = self.node_indices[node]

        # Use the previously computed left_stack of the node just before `node` as the base, if it exists.
        previous_stack = self.left_stacks[self.main_nodes[index - 1]] if index > 0 else None
        contracted = node.contract_vertically()
        if previous_stack is not None:
            connect_labels = contracted.get_connecting_labels(previous_stack)
            contracted = previous_stack.contract_with(contracted, connect_labels)
        self.left_stacks[node] = contracted

    def right_update_stacks(self, node):
        """Updates the right stacks."""
        index = self.node_indices[node]

        # Use the previously computed right_stack of the node just after `node` as the base, if it exists.
        previous_stack = self.right_stacks[self.main_nodes[index + 1]] if index < len(self.main_nodes) - 1 else None
        contracted = node.contract_vertically()
        if previous_stack is not None:
            connect_labels = previous_stack.get_connecting_labels(contracted)
            contracted = contracted.contract_with(previous_stack, connect_labels)
        self.right_stacks[node] = contracted

    def step(self, node, d_loss, dd_loss, lr=1.0, method='exact', eps=0.0, clip_value=None):
        """Finds the update step for a given node"""

        # Determine broadcast
        broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
        non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)
        
        # Compute the Jacobian
        J = self.compute_jacobian_stack(node).expand_labels(self.output_labels, d_loss.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        all_letters = iter(string.ascii_letters)
        dim_labels = {dim: next(all_letters) for dim in self.output_labels}  # Assign letters to output dims
        for d in non_broadcast_dims:
            dim_labels['_' + d] = next(all_letters)
        
        dd_loss_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels[d] for d in non_broadcast_dims] + [dim_labels['_' + d] for d in non_broadcast_dims])
        d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)

        J_ein1 = ''
        J_ein2 = ''
        J_out1 = ''
        J_out2 = ''
        for d in J.dim_labels:
            if d not in dim_labels and d not in broadcast_dims:
                dim_labels[d] = next(all_letters)
                dim_labels['_' + d] = next(all_letters)
            J_ein1 += dim_labels[d]
            J_ein2 += dim_labels['_' + d] if d not in broadcast_dims else dim_labels[d]
            if d not in broadcast_dims:
                J_out1 += dim_labels[d]
                J_out2 += dim_labels['_' + d]

        # Construct einsum notations
        einsum_A = f'{J_ein1},{J_ein2},{dd_loss_ein}->{J_out1}{J_out2}'
        einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"

        # Compute einsum operations
        A = torch.einsum(einsum_A, J.tensor, J.tensor, dd_loss)
        b = torch.einsum(einsum_b, J.tensor, d_loss)

        # Solve the system
        A_f = A.flatten(0, A.ndim//2-1).flatten(1, -1)
        b_f = b.flatten()
        
        scale = A_f.abs().max().clamp(min=1e-12)
        A_f = A_f / scale
        b_f = b_f / scale

        if method == 'exact':
            x = torch.linalg.solve(A_f, -b_f)
        elif method == 'cholesky':
            A_f_reg = A_f + eps * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
            L = torch.linalg.cholesky(A_f_reg)
            x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
            x = x.squeeze(-1)
        elif method.lower() == 'tt' or method.lower() == 'tensor_train':
            x = tensor_gradient_solver(J, d_loss, dd_loss, broadcast_dims, r=2, ridge_eps=eps).permute(*node.dim_labels)
            return x.tensor
        else:
            raise ValueError(f"Unknown method: {method}")

        step_tensor = x.reshape(b.shape)

        # Permute to match node dimension order
        permute = [J.dim_labels[len(broadcast_dims):].index(l) for l in node.dim_labels]
        step_tensor = step_tensor.permute(*permute)

        node.update_node(step_tensor, lr=lr)
        return step_tensor

    def get_A_b(self, node, d_loss, sqd_loss, output_dims=tuple()):
        """Finds the update step for a given node"""

        # Determine broadcast and gradient dimensions
        broadcast_dims = tuple(b for b in output_dims if b not in node.dim_labels)
        J = self.compute_jacobian_stack(node).expand_labels(output_dims, d_loss.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        all_letters = iter(string.ascii_letters)
        dim_labels = {dim: next(all_letters) for dim in output_dims}  # Assign letters to output dims
        broad_ein = ''.join(dim_labels[b] for b in broadcast_dims)
        output_ein = ''.join(dim_labels[o] for o in output_dims)

        # Generate jacobian dimension labels
        num_jacobian_dims = len(J.dim_labels) - len(broadcast_dims)
        grad1_ein = ''.join(next(all_letters) for _ in range(num_jacobian_dims))
        grad2_ein = ''.join(next(all_letters) for _ in range(num_jacobian_dims))

        # Construct einsum notations
        einsum_A = f"{broad_ein}{grad1_ein},{broad_ein}{grad2_ein},{output_ein}->{grad1_ein}{grad2_ein}"
        einsum_b = f"{broad_ein}{grad1_ein},{output_ein}->{grad1_ein}"

        # Compute einsum operations
        A = torch.einsum(einsum_A, J.tensor, J.tensor, sqd_loss)
        b = torch.einsum(einsum_b, J.tensor, d_loss)

        return A, b, J

    def set_input(self, x):
        """Sets the input tensor for the network."""
        was_updated = False
        if isinstance(x, tuple) or isinstance(x, list):
            for node, tensor in zip(self.input_nodes, x):
                if node.tensor is not tensor:
                    was_updated = True
                node.set_tensor(tensor)
        else:
            for node in self.input_nodes:
                if node.tensor is not x:
                    was_updated = True
                node.set_tensor(x)
        if was_updated:
            self.left_stacks = None
            self.right_stacks = None
            print("Input tensors updated. Stacks will be recomputed.")
        return was_updated
    
    def disconnect(self, nodes):
        """Disconnects the given node from the network."""
        if not isinstance(nodes, list) and not isinstance(nodes, tuple):
            nodes = [nodes]
        for n_rem in nodes:
            for n in self.nodes:
                if n_rem in n.connections.values():
                    for l in n.get_connecting_labels(n_rem):
                        del n_rem.connections[l]
                        del n.connections[l]
            self.nodes.remove(n_rem)

    def swipe(self, x, y_true, loss_fn, method='exact', eps=1e-12, num_swipes=1, lr=1.0, convergence_criterion=None, orthonormalize=False, verbose=False, skip_right=False):
        """Swipes the network to minimize the loss function."""
        y_pred = self.forward(x).permute_first(*self.output_labels).tensor
        loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_true)
        if verbose:
            print('Initial loss:', loss.mean().item())
        converged = False
        for _ in range(num_swipes):
            for n in self.main_nodes:
                self.step(n, d_loss, sqd_loss, lr=lr, method=method, eps=eps)
                if orthonormalize:
                    self.node_orthonormalize_left(n)
                self.left_update_stacks(n)

                y_pred = self.forward(x).permute_first(*self.output_labels).tensor

                loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_true)
                if verbose:
                    print('Full loss:', loss.mean().item())
                if convergence_criterion is not None and convergence_criterion(y_pred, y_true):
                    print('Converged')
                    converged = True
                    break
            if converged:
                break
            if skip_right:
                continue
            for n in reversed(self.main_nodes):
                self.step(n, d_loss, sqd_loss, lr=lr, method=method, eps=eps)
                if orthonormalize:
                    self.node_orthonormalize_right(n)
                self.right_update_stacks(n)
                
                y_pred = self.forward(x).permute_first(*self.output_labels).tensor

                loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_true)
                if verbose:
                    print('Full loss:', loss.mean().item())
                if convergence_criterion is not None and convergence_criterion(y_pred, y_true):
                    print('Converged')
                    converged = True
                    break
            if converged:
                break

    def orthonormalize_left(self):
        """
        Orthonormalizes the left bonds of the main nodes using QR decomposition.
        """
        for n in self.main_nodes:
            self.node_orthonormalize_left(n)

    def orthonormalize_right(self):
        """
        Orthonormalizes the right bonds of the main nodes using QR decomposition.
        """
        for n in self.main_nodes:
            self.node_orthonormalize_right(n)

    def node_orthonormalize_left(self, node):
        index = self.node_indices[node]
        if index >= len(self.main_nodes) - 1:
            return

        # Get bond dimension indices
        right_bond_indices = [node.dim_labels.index(lbl) for lbl in node.right_labels]
        all_indices = list(range(len(node.shape)))
        left_nonbond_indices = [idx for idx in all_indices if idx not in right_bond_indices]

        # Permute so that left dims are first, right bond last
        perm_order = left_nonbond_indices + right_bond_indices
        A_permuted = node.tensor.permute(perm_order)

        # Flatten into a 2D matrix: (left combined, right bond)
        original_shape = A_permuted.shape
        A_reshaped = A_permuted.flatten(0, len(left_nonbond_indices)-1)

        # Perform QR decomposition
        Q, R = torch.linalg.qr(A_reshaped, mode='reduced')

        # Reshape Q back
        Q = Q.reshape(original_shape[:len(left_nonbond_indices)] + (Q.shape[-1],))

        # Reverse the permutation
        inverse_perm = sorted(range(len(perm_order)), key=lambda k: perm_order[k])
        node.tensor = Q.permute(*inverse_perm)

        # Push R into the next node's left bonds
        next_node = self.main_nodes[index + 1]
        bond_dims = node.get_connecting_labels(next_node)
        next_node.permute_first(*bond_dims)
        next_node.tensor = torch.einsum('ij,j...->i...', R, next_node.tensor)

        if self.right_stacks is not None:
            self.right_update_stacks(next_node)

    def node_orthonormalize_right(self, node):
        index = self.node_indices[node]
        if index <= 0:
            return

        # Identify left bond indices and the rest.
        left_bond_indices = [node.dim_labels.index(lbl) for lbl in node.left_labels]
        all_indices = list(range(len(node.shape)))
        right_nonbond_indices = [idx for idx in all_indices if idx not in left_bond_indices]

        # Permute so that left bonds come last (we flip the order to simulate RQ)
        # One strategy is to bring the nonbond dims first, then the bond dims.
        perm_order = right_nonbond_indices + left_bond_indices
        A_permuted = node.tensor.permute(perm_order)

        # Save original shape to later reshape Q back.
        original_shape = A_permuted.shape

        # Flatten: rows combine nonbond dims, columns combine left bond dims.
        A_reshaped = A_permuted.flatten(0, len(right_nonbond_indices)-1)

        # --- Simulate RQ decomposition ---
        # Reverse both dimensions.
        A_rev = torch.flip(A_reshaped, dims=[0, 1])
        Q_rev, R_rev = torch.linalg.qr(A_rev, mode='reduced')
        # Reverse factors back
        R = torch.flip(R_rev.T, dims=[0, 1])
        Q = torch.flip(Q_rev, dims=[0, 1])
        # --- End RQ ---

        # Reshape Q back to tensor form.
        Q = Q.reshape(original_shape[:len(right_nonbond_indices)] + (Q.shape[-1],))

        # Reverse the permutation
        inverse_perm = sorted(range(len(perm_order)), key=lambda k: perm_order[k])
        node.tensor = Q.permute(*inverse_perm)

        # Push R into the previous node's right bonds
        prev_node = self.main_nodes[index - 1]

        bond_dims = node.get_connecting_labels(prev_node)
        prev_node.permute_last(*bond_dims)
        prev_node.tensor = torch.einsum('ji,...j->...i', R, prev_node.tensor)

        if self.left_stacks is not None:
            self.left_update_stacks(prev_node)