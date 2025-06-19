#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tensor.bregman import SquareComplexBregFunction
from tensor.layers import ComplexTensorTrainLayer
from tqdm.auto import tqdm
torch.set_default_dtype(torch.float64)
p = 5
degree = 3
coeffs = torch.randn(degree, 1, p)
# def func(x):
#     prod = ((x**3).sum(dim=-1, keepdim=True) + coeffs[0])
#     for i in range(degree-1):
#         prod = prod * ((x**3).sum(dim=-1, keepdim=True) + coeffs[i+1])
#     return prod.sum(-1, keepdim=True)
def func(x):
    prod = (x + coeffs[0])
    for i in range(degree-1):
        prod = prod * (x + coeffs[i+1])
    return prod.sum(dim=-1, keepdim=True)
x_train = torch.sort((torch.rand(10000, p)-0.5)*2).values
y_train = func(x_train).cuda()
x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).cuda()


x_test = torch.sort((torch.rand(1000, p)-0.5)*2).values
y_test = func(x_test).cuda()
x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).cuda()
#%%
# Convert everything to complex
x_train = x_train.to(dtype=torch.complex128)
y_train = y_train.to(dtype=torch.complex128)
x_test = x_test.to(dtype=torch.complex128)
y_test = y_test.to(dtype=torch.complex128)
#%%
N = 8
r = 6
method = 'exact'
# Define Bregman function
bf = SquareComplexBregFunction()
layer = ComplexTensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=False).cuda()
y_pred_prev = layer(x_train)
#%%
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    return False
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=-1, lr=1.0, eps=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=5, skip_second=False, direction='l2r', disable_tqdm=True)
#%%
