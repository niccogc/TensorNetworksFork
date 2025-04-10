#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
torch.set_default_dtype(torch.float64)
from tensor_layers import TensorOperatorLayer
from BregmanLoss import SquareBregFunction

#%%
r = 5
p = 6
N = 3
D = torch.zeros((p,p,p))
for i in range(p):
    for j in range(p):
        for k in range(p):
            D[i,j,k] = 1 if (k-(i+j)) >= 0 else 0
I = torch.eye(p)
OK = torch.einsum('ijk,jl->ijlk', D, I)
ops = []
for n in range(N):
    if n == 0:
        O = OK[:1]
    elif n == N-1:
        O = OK[..., -1:]
    else:
        O = OK
    ops.append(O)
#%%
layer = TensorOperatorLayer(OK, p, r, N).cuda()
#%%
# Create input data
S = 10
x = torch.randn(S, N)
xinp = torch.pow(x.unsqueeze(-1), torch.arange(p).unsqueeze(0).unsqueeze(0)).cuda()

# Define Bregman function
bf = SquareBregFunction()

#y = func3(x)
y = (x.sum(-1, keepdim=True))**2

y = y.cuda()
xinp = xinp.cuda()
xinput = [xinp[:, i] for i in range(len(layer.tensor_network.input_nodes))]

with torch.inference_mode():
    layer.tensor_network.swipe(xinput, y, bf, verbose=True)
#%%
layer.tensor_network.disconnect(layer.tensor_network.input_nodes) # Make this one virtual by returning a virtual tensor network
#%%
layer.tensor_network.recompute_all_stacks()
out = layer.tensor_network.forward(xinput)
# %%
print(out.tensor[*out.tensor.nonzero(as_tuple=True)])
print(out.tensor.nonzero())
print(out.tensor.nonzero().sum(1).max())
#%%