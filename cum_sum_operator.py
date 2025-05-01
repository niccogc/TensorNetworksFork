#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
torch.set_default_dtype(torch.float64)
from tensor.layers import TensorOperatorLayer
from tensor.bregman import SquareBregFunction
from tensor.utils import visualize_tensornetwork

#%%
r = 5
p = 6
N = 3

ops = []
for n in range(N):
    H = torch.triu(torch.ones((1 if n == 0 else p, p), dtype=torch.float64), diagonal=0)
    D = torch.zeros((p, p, p, 1 if n == N-1 else p), dtype=torch.float64)
    for i in range(p):
        D[i, i, i, 0 if n == N-1 else i] = 1
    C = torch.einsum('ij,j...->i...', H, D)
    ops.append(C)
#%%
layer = TensorOperatorLayer(ops, p, r, N).cuda()
#%%
# Create input data
S = 10
x = torch.randn(S, p-1)
xinp = torch.concat((x, torch.ones(S, 1)), dim=-1)

# Define Bregman function
bf = SquareBregFunction()

#y = func3(x)
y = (x.sum(-1, keepdim=True))**2

y = y.cuda()
xinp = xinp.cuda()

with torch.inference_mode():
    layer.tensor_network.accumulating_swipe(xinp, y, bf, batch_size=-1, verbose=True)
#%%
train_network = layer.tensor_network.disconnect(layer.tensor_network.input_nodes) # Make this one virtual by returning a virtual tensor network
#%%
train_network.recompute_all_stacks()
out = train_network.forward(xinp)
# %%
print(out.tensor[*out.tensor.nonzero(as_tuple=True)])
print(out.tensor.nonzero())
print(out.tensor.nonzero().sum(1).max())
#%%

# Visualize the tensor network
def visualize_cum_sum_operator():
    visualize_tensornetwork(layer.tensor_network, layout='grid')

visualize_cum_sum_operator()
#%%