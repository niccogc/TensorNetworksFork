import torch
from tensor.node import TensorNode
import string

def tensor_gradient_solver(J, d_loss, sqd_loss, broadcast_dims, r=2, ridge_eps=0.0):
    """Solves for the gradient using tensor decomposition."""
    bond_labels = [l for l in J.dim_labels if l not in broadcast_dims]
    x_blocks = []
    for i, l in enumerate(bond_labels):
        x_block = TensorNode((1 if i == 0 else r, J.dim_size(l), 1 if i == len(bond_labels) - 1 else r), [f'<TB>{i}', l, f'<TB>{i+1}'], name=f'X{i}')
        x_blocks.append(x_block)
        x_block.squeeze()

    for i, block in enumerate(x_blocks):
        # Precalculate blocks
        X_gradient = J
        for b2 in x_blocks:
            if b2 != block:
                X_gradient = X_gradient.contract_with(b2, list(set(X_gradient.dim_labels).intersection(b2.dim_labels)))
        JJ = J.contract_with(J, [l for l in J.dim_labels if l not in broadcast_dims])
        
        # Create dummy dimensions and define order of labels
        right_label_order = [l for l in X_gradient.dim_labels if l not in broadcast_dims]
        JJ_gradient = JJ.contract_with(X_gradient, []).permute_first(*right_label_order)
        left_label_order = [f"_{l}" for l in right_label_order]

        # Calculate the left-hand side
        X_gradient_transpose = X_gradient.get_transposed_node(set(broadcast_dims))
        A_side = JJ_gradient.contract_with(X_gradient_transpose, [])
        A_side.permute_first(*left_label_order, *right_label_order)

        A = A_side.tensor
        b = JJ_gradient.tensor

        # Einsum with the loss derivatives
        broad_dims = string.ascii_letters[:len(broadcast_dims)]
        A_ein_str = f'...{broad_dims},{broad_dims}->...'
        b_ein_str = f'...{broad_dims},{broad_dims},{broad_dims}->...'
        A = torch.einsum(A_ein_str, A, sqd_loss)
        b = torch.einsum(b_ein_str, b, d_loss, sqd_loss)

        original_shape = b.shape

        A_f = A.flatten(0, A.ndim//2-1).flatten(1, -1)
        b_f = b.flatten()

        A_f = A_f + ridge_eps * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)

        x = torch.linalg.solve(A_f, b_f)
        x_blocks[i] = TensorNode(x.reshape(original_shape), right_label_order, name=f'X{i}')
        x_blocks[i].squeeze()

    # Contract each block
    contracted = x_blocks[0]
    for block in x_blocks[1:]:
        contracted = contracted.contract_with(block, list(set(contracted.dim_labels).intersection(block.dim_labels)))
    return contracted