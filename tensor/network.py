import torch
import string
from collections import deque
from tensor.node import TensorNode
from tqdm.auto import tqdm
import time

class TensorNetwork:
    def __init__(self, input_nodes, main_nodes, train_nodes=None, output_labels=('s',), sample_dim='s'):
        """Initializes a TensorNetwork with ordered input and main nodes."""
        self.input_nodes = input_nodes
        self.main_nodes = main_nodes
        self.train_nodes = main_nodes if train_nodes is None else train_nodes
        #self.node_indices = {node: i for i, node in enumerate(main_nodes)}
        self.left_stacks = None
        self.right_stacks = None
        self.output_labels = output_labels
        self.sample_dim = sample_dim
        self.nodes, self.node_indices = self._discover_nodes()
    
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
        node_indices = {node: i for i, node in enumerate(self.main_nodes)}
        discovered = set(self.main_nodes)
        queue = deque(self.main_nodes)

        while queue:
            node = queue.popleft()
            current_index = node_indices[node]
            for label, connected_node in node.connections.items():
                if connected_node not in discovered and not node.is_horizontal_bond(label):
                    discovered.add(connected_node)
                    queue.append(connected_node)
                    node_indices[connected_node] = current_index

        return list(sorted(discovered, key=lambda n: n.name)), node_indices

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

    @torch.no_grad()
    def get_A_b(self, node, grad, hessian):
        """Finds the update step for a given node"""

        # Determine broadcast
        broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
        non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)
        
        # Compute the Jacobian
        J = self.compute_jacobian_stack(node).expand_labels(self.output_labels, grad.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        all_letters = iter(string.ascii_letters)
        dim_labels = {dim: next(all_letters) for dim in self.output_labels}  # Assign letters to output dims
        for d in non_broadcast_dims:
            dim_labels['_' + d] = next(all_letters)
        
        dd_loss_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels[d] for d in non_broadcast_dims] + [dim_labels['_' + d] for d in non_broadcast_dims])
        d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)

        J_ein1 = ''
        J_ein2 = ''
        J_out1 = []
        J_out2 = []
        dim_order = []
        for d in J.dim_labels:
            if d not in dim_labels and d not in broadcast_dims:
                dim_labels[d] = next(all_letters)
                dim_labels['_' + d] = next(all_letters)
            J_ein1 += dim_labels[d]
            J_ein2 += dim_labels['_' + d] if d != self.sample_dim else dim_labels[d]
            if d not in broadcast_dims:
                J_out1.append(dim_labels[d])
                J_out2.append(dim_labels['_' + d])
                dim_order.append(d)
                
        J_out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])
        J_out2 = ''.join([J_out2[dim_order.index(d)] for d in node.dim_labels])

        # Construct einsum notations
        einsum_A = f'{J_ein1},{J_ein2},{dd_loss_ein}->{J_out1}{J_out2}'
        einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"
        
        # Compute einsum operations
        A = torch.einsum(einsum_A, J.tensor, J.tensor, hessian)
        b = torch.einsum(einsum_b, J.tensor, grad)

        J_sum = J.permute_first(dim_order).sum_labels(broadcast_dims)
        return A, b, J_sum.flatten()
    
    def solve_system(self, node, A, b, J, method='exact', eps=0.0, delta=1e2):
        """Finds the update step for a given node"""
        # Solve the system
        A_f = A.flatten(0, A.ndim//2-1).flatten(1, -1)
        b_f = b.flatten()

        scale = A_f.diag().abs().mean()
        A_f = A_f / scale
        b_f = b_f / scale

        if method.lower() == 'exact':
            x = torch.linalg.solve(A_f, -b_f)
        elif method.lower() == 'ridge_exact':
            A_f.add_((2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device))
            b_f.add_((2 * eps) * node.tensor.flatten())
            x = torch.linalg.solve(A_f, -b_f)
        elif method.lower() == 'cholesky':
            A_f_reg = A_f + eps * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
            L = torch.linalg.cholesky(A_f_reg)
            x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
            x = x.squeeze(-1)
        elif method.lower().startswith('dog'):
            # Dogleg trust-region method
            # Steepest descent step: p_sd = - (g^T g) / (g^T A g) * g
            p_sd = -b_f

            # Gauss-Newton step: solve A p_gn = -g
            try:
                p_gn = torch.linalg.solve(A_f, p_sd)
            except torch.linalg.LinAlgError:
                # Fallback to steepest descent if singular
                print("It was singular")
                p_gn = p_sd

            norm_p_gn = torch.norm(p_gn)
            norm_p_sd = torch.norm(p_sd)
            tau = (norm_p_sd ** 2) / (torch.norm(J * p_sd) ** 2)
            t = tau * p_sd
            norm_t = torch.norm(t)

            if norm_p_gn <= delta:
                x = p_gn
                print("I did a gauss newton step! :D")
            elif norm_t >= delta:
                x = (delta / norm_p_sd) * p_sd
                print("I did a steepest descent step! :(")
            else:
                d = p_gn - t

                a = torch.norm(d)
                b_coeff = 2 * torch.dot(t, d)
                c = norm_t - delta**2
                
                # Solve for beta, choose the one that is between 0 and 1
                beta_p = (-b_coeff + torch.sqrt(b_coeff**2 - 4 * a * c)) / (2 * a)
                beta_n = (-b_coeff - torch.sqrt(b_coeff**2 - 4 * a * c)) / (2 * a)
                if beta_p >= 0 and beta_p <= 1:
                    beta = beta_p
                elif beta_n >= 0 and beta_n <= 1:
                    beta = beta_n
                elif beta_p > 1:
                    beta = beta_p
                elif beta_n > 1:
                    beta = beta_n
                else:
                    raise ValueError("No valid beta found in dogleg method.")
                x = t + beta * d
                print(f"Beta: {beta.item():.4g}, Tau: {tau.item():.4g} :)))")

        else:
            raise ValueError(f"Unknown method: {method}")

        step_tensor = x.reshape(b.shape)
        return step_tensor


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
        return was_updated
    
    def disconnect(self, nodes):
        """Creates a new TensorNetwork without the given nodes and their connections.
        The nodes themselves are virtual copies with their own connections dictionary.
        Only the underlying tensors are carried over."""
        if not isinstance(nodes, list) and not isinstance(nodes, tuple):
            nodes = [nodes]

        # Create a mapping of old nodes to new virtual copies
        node_mapping = {}
        for node in self.nodes:
            if node not in nodes:
                # Create a virtual copy of the node with the same tensor
                new_node = TensorNode(node.tensor, node.dim_labels, l=node.left_labels, r=node.right_labels, name=node.name)
                node_mapping[node] = new_node

        # Recreate connections in the new virtual nodes
        for old_node, new_node in node_mapping.items():
            for label, connected_node in old_node.connections.items():
                if connected_node in node_mapping:  # Only add connections to nodes that are not being removed
                    new_node.connections[label] = node_mapping[connected_node]
                    new_node.connection_priority[label] = old_node.connection_priority[label]

        # Create new input and main nodes lists
        new_input_nodes = [node_mapping[node] for node in self.input_nodes if node in node_mapping]
        new_main_nodes = [node_mapping[node] for node in self.main_nodes if node in node_mapping]
        new_train_nodes = [node_mapping[node] for node in self.train_nodes if node in node_mapping]

        # Create a new TensorNetwork object
        new_network = TensorNetwork(new_input_nodes, new_main_nodes, new_train_nodes, self.output_labels, self.sample_dim)

        return new_network

    def accumulating_swipe(self, x, y_true, loss_fn, batch_size=1, num_swipes=1, lr=1.0, method='exact', eps=1e-12, delta=1.0, convergence_criterion=None, orthonormalize=False, verbose=False, skip_right=False, timeout=None):
        """Swipes the network to minimize the loss using accumulated A and b over mini-batches.
        Args:
            timeout (float or None): Maximum time in seconds to run. If None, no timeout.
        """
        data_size = len(x) if isinstance(x, torch.Tensor) else x[0].shape[0]
        if batch_size <= 0:
            batch_size = data_size
        batches = (data_size + batch_size - 1) // batch_size # round up division

        node_l2r = None
        node_r2l = None
        start_time = time.time() if timeout is not None else None

        for NS in range(num_swipes):
            if isinstance(eps, list):
                eps_ = eps[NS*2]
            else:
                eps_ = eps
            # LEFT TO RIGHT
            for node_l2r in self.train_nodes:
                if node_l2r in self.node_indices and node_r2l in self.node_indices and self.node_indices[node_l2r] == self.node_indices[node_r2l]:
                    continue
                # Timeout check
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                    return False
                A_out, b_out, J_out = None, None, None
                total_loss = 0.0

                for b in tqdm(range(batches), desc=f"Left to right pass ({node_l2r.name if hasattr(node_l2r, 'name') else 'node'})", disable=verbose < 2):
                    # Timeout check inside batch loop
                    if timeout is not None and (time.time() - start_time) > timeout:
                        print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                        return False
                    x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                    y_batch = y_true[b*batch_size:(b+1)*batch_size]

                    y_pred = self.forward(x_batch).permute_first(*self.output_labels).tensor
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)

                    A, b_vec, J = self.get_A_b(node_l2r, d_loss, sqd_loss)
                    if A_out is None:
                        A_out = A
                        b_out = b_vec
                        J_out = J
                    else:
                        A_out.add_(A)
                        b_out.add_(b_vec)
                        J_out.add_(J)

                    total_loss += loss.mean().item()

                if verbose:
                    print(f"NS: {NS}, Left loss ({node_l2r.name if hasattr(node_l2r, 'name') else 'node'}):", total_loss / batches, f" (eps: {eps_})")
                try:
                    step_tensor = self.solve_system(node_l2r, A_out, b_out, J_out, method=method, eps=eps_, delta=delta)
                except torch.linalg.LinAlgError:
                    print(f"Singular system for node {node_l2r.name if hasattr(node_l2r, 'name') else 'node'}")
                    return False
                
                node_l2r.update_node(step_tensor, lr=lr)
                if orthonormalize:
                    self.node_orthonormalize_left(node_l2r)
                self.left_update_stacks(node_l2r)

                # Convergence check after node update (pause timer here)
                if convergence_criterion is not None:
                    pause_time = time.time() if timeout is not None else None
                    y_trues = []
                    y_preds = []
                    for b in range(batches):
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        y_batch = y_true[b*batch_size:(b+1)*batch_size]

                        y_pred = self.forward(x_batch).permute_first(*self.output_labels).tensor
                        y_trues.append(y_batch)
                        y_preds.append(y_pred)
                    y_trues = torch.cat(y_trues, dim=0)
                    y_preds = torch.cat(y_preds, dim=0)

                    if timeout is not None:
                        # Resume timer after convergence check
                        start_time += time.time() - pause_time

                    if convergence_criterion(y_preds, y_trues):
                        print('Converged (left pass)')
                        return True

            if skip_right:
                continue

            if isinstance(eps, list):
                eps_ = eps[NS*2+1]
            else:
                eps_ = eps

            # RIGHT TO LEFT
            for node_r2l in reversed(self.train_nodes):
                if node_l2r in self.node_indices and node_r2l in self.node_indices and self.node_indices[node_l2r] == self.node_indices[node_r2l]:
                    continue
                # Timeout check
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                    return False
                A_out, b_out, J_out = None, None, None
                total_loss = 0.0

                for b in tqdm(range(batches), desc=f"Right to left pass ({node_r2l.name if hasattr(node_r2l, 'name') else 'node'})", disable=verbose < 2):
                    # Timeout check inside batch loop
                    if timeout is not None and (time.time() - start_time) > timeout:
                        print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                        return False
                    x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                    y_batch = y_true[b*batch_size:(b+1)*batch_size]

                    y_pred = self.forward(x_batch).permute_first(*self.output_labels).tensor
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)

                    A, b_vec, J = self.get_A_b(node_r2l, d_loss, sqd_loss)
                    if A_out is None:
                        A_out = A
                        b_out = b_vec
                        J_out = J
                    else:
                        A_out.add_(A)
                        b_out.add_(b_vec)
                        J_out.add_(J)

                    total_loss += loss.mean().item()

                if verbose:
                    print(f"NS: {NS}, Right loss ({node_r2l.name if hasattr(node_r2l, 'name') else 'node'}):", total_loss / batches, f" (eps: {eps_})")
                try:
                    step_tensor = self.solve_system(node_r2l, A_out, b_out, J_out, method=method, eps=eps_, delta=delta)
                except torch.linalg.LinAlgError:
                    print(f"Singular system for node {node_r2l.name if hasattr(node_r2l, 'name') else 'node'}")
                    return False
                
                node_r2l.update_node(step_tensor, lr=lr)
                if orthonormalize:
                    self.node_orthonormalize_right(node_r2l)
                self.right_update_stacks(node_r2l)

                # Convergence check after node update (pause timer here)
                if convergence_criterion is not None:
                    pause_time = time.time() if timeout is not None else None
                    y_trues = []
                    y_preds = []
                    for b in range(batches):
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        y_batch = y_true[b*batch_size:(b+1)*batch_size]

                        y_pred = self.forward(x_batch).permute_first(*self.output_labels).tensor
                        y_trues.append(y_batch)
                        y_preds.append(y_pred)
                    y_trues = torch.cat(y_trues, dim=0)
                    y_preds = torch.cat(y_preds, dim=0)

                    if timeout is not None:
                        # Resume timer after convergence check
                        start_time += time.time() - pause_time

                    if convergence_criterion(y_preds, y_trues):
                        print('Converged (right pass)')
                        return True

        return False

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