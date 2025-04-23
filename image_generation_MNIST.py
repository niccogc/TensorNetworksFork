#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch.set_default_dtype(torch.float64)
from tensor.node import TensorNode
from tensor.network import TensorNetwork
from tensor.layers import TensorNetworkLayer

A1 = TensorNode((1, 784, 3, 11), ['b0', 'w', 'b1', 'p'], name='A1', l='b0', r='b1')
A2 = TensorNode((3, 1, 40, 11), ['b1', 'c', 'b2', 'p'], name='A2', l='b1', r='b2')
A3 = TensorNode((40, 1, 1, 11), ['b2', 'h', 'b3', 'p'], name='A3', l='b2', r='b3')
X1 = TensorNode((1, 11), ['s', 'p'], name='X1')
X2 = TensorNode((1, 11), ['s', 'p'], name='X2')
X3 = TensorNode((1, 11), ['s', 'p'], name='X3')

A1.connect(A2, 'b1')
A2.connect(A3, 'b2')
X1.connect(A1, 'p')
X2.connect(A2, 'p')
X3.connect(A3, 'p')

for a in [A1, A2, A3]:
    a.squeeze(exclude={'s'})
for x in [X1, X2, X3]:
    x.squeeze(exclude={'s'})

# Create a TensorNetwork
tensor_net = TensorNetwork([X1, X2, X3], [A1, A2, A3])
#tensor_net.orthonormalize_left()
network = TensorNetworkLayer(tensor_net, ('s', 'w')).cuda()
#%%
import torch
images = torch.load("./latents/test_images.pt").squeeze(1).cuda()
latents = torch.load("./latents/test_latents.pt").cuda()
labels = torch.load("./latents/test_labels.pt").cuda()
# %%
import torch
from tensor.bregman import Square2DBregFunction
data = torch.cat([torch.ones(latents.size(0), 1, device=latents.device), latents], dim=-1)

# Define Bregman function
bf = Square2DBregFunction()

method = 'cholesky'
output_labels = ('s', 'w')
lr = 1.0
eps = 1e-3
batch_size = [16, 8, 8]
#%%
from tqdm import tqdm
# LEFT LOSS
with torch.inference_mode():
    for n, bs in zip(tensor_net.main_nodes, batch_size):
        s = bs
        batches = data.size(0) // s
        A_out = None
        b_out = None
        full_loss = 0.0

        for b in tqdm(range(batches)):
            xinp = data[b*s:(b+1)*s]
            y = images[b*s:(b+1)*s].reshape(s, 784)
            if len(xinp) < s:
                break
            tensor_net.set_node(n) # just for better stability
            x_pred = tensor_net.forward(xinp).permute_first(*output_labels).tensor
            loss, d_loss, sqd_loss = bf.forward(x_pred, y)
            A_f, b_f, J = tensor_net.get_A_b(n, d_loss, sqd_loss, output_labels)
            if A_out is None:
                A_out = A_f
                b_out = b_f
            else:
                A_out.add_(A_f)
                b_out.add_(b_f)
            full_loss += loss.mean().item()

        print("Full loss before: ", full_loss / batches)

        A_f = A_out.flatten(0, A_out.ndim//2-1).flatten(1, -1)
        b_f = b_out.flatten()

        A_f_reg = A_f + eps * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)

        L = torch.linalg.cholesky(A_f_reg)
        x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
        x = x.squeeze(-1)

        step_tensor = x.reshape(b_out.shape)

        # Permute to match node dimension order
        broadcast_dims = tuple(b for b in output_labels if b not in n.dim_labels)
        permute = [J.dim_labels[len(broadcast_dims):].index(l) for l in n.dim_labels]
        step_tensor = step_tensor.permute(*permute)

        n.update_node(step_tensor, lr=lr)
        tensor_net.node_orthonormalize_left(n)
        tensor_net.left_update_stacks(n, orthonormalize=True)
    for n, bs in reversed(list(zip(tensor_net.main_nodes, batch_size))):
        s = bs
        batches = data.size(0) // s
        A_out = None
        b_out = None
        full_loss = 0.0

        for b in tqdm(range(batches)):
            xinp = data[b*s:(b+1)*s]
            y = images[b*s:(b+1)*s].reshape(-1, 784)
            if len(xinp) < s:
                break
            tensor_net.set_node(n) # just for better stability
            x_pred = tensor_net.forward(xinp).permute_first(*output_labels).tensor
            loss, d_loss, sqd_loss = bf.forward(x_pred, y)
            A_f, b_f, J = tensor_net.get_A_b(n, d_loss, sqd_loss, output_labels)
            if A_out is None:
                A_out = A_f
                b_out = b_f
            else:
                A_out.add_(A_f)
                b_out.add_(b_f)
            full_loss += loss.mean().item()

        print("Full loss before: ", full_loss / batches)

        A_f = A_out.flatten(0, A_out.ndim//2-1).flatten(1, -1)
        b_f = b_out.flatten()

        A_f_reg = A_f + eps * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)

        L = torch.linalg.cholesky(A_f_reg)
        x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
        x = x.squeeze(-1)

        step_tensor = x.reshape(b_out.shape)

        # Permute to match node dimension order
        broadcast_dims = tuple(b for b in output_labels if b not in n.dim_labels)
        permute = [J.dim_labels[len(broadcast_dims):].index(l) for l in n.dim_labels]
        step_tensor = step_tensor.permute(*permute)

        n.update_node(step_tensor, lr=lr)
        tensor_net.node_orthonormalize_right(n)
        tensor_net.right_update_stacks(n, orthonormalize=True)
# %%
# Try to reconstruct some unused latents (10)
import matplotlib.pyplot as plt
with torch.inference_mode():
    images = torch.load("./latents/test_images.pt")[-10:].squeeze(1).cuda()
    latents = torch.load("./latents/test_latents.pt")[-10:].cuda()
    labels = torch.load("./latents/test_labels.pt")[-10:].cuda()
    latents0 = torch.cat([torch.ones(latents.size(0), 1, device=latents.device), latents], dim=-1)
    images_pred = network(latents0)
    images_pred = images_pred.reshape(-1, 28, 28)
    
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(images_pred[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i+11)
        plt.imshow(images[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# %%
from sklearn.decomposition import PCA
# PCA of latents
latents = torch.load("./latents/test_latents.pt").numpy()
labels = torch.load("./latents/test_labels.pt").numpy()
pca = PCA(n_components=2)
latents_pca = pca.fit_transform(latents)
plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=labels, cmap='tab10')
plt.show()
# %%
