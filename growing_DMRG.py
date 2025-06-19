#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainDMRGInfiLayer
from tqdm.auto import tqdm
torch.set_default_dtype(torch.float64)
p = 4
degree = 6
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
r = 10
carts = 2
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    global carts
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    train_loss_dict[carts] = rmse.item()
    #print(carts, 'Train RMSE:', rmse.item())
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test - y_test)**2))
    test_loss_dict[carts] = rmse.item()
    return False
method = 'ridge_cholesky'
# Define Bregman function
split_errors = []
bf = SquareBregFunction()
layer = TensorTrainDMRGInfiLayer(r, x_train.shape[1], output_shape=1, constrict_bond=True).cuda()
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=-1, lr=1.0, eps=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=5, skip_second=False, direction='l2r', disable_tqdm=True)
#%%
total_carts = 10
for carts in tqdm(range(2+1, total_carts+1)):
    layer.grow_middle()
    layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=-1, lr=1.0, eps=1e-3, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=5, skip_second=False, direction='l2r', disable_tqdm=True)
    node = layer.nodes[layer.num_carriages//2]
    left_labels = node.dim_labels[:2]
    right_labels = node.dim_labels[-2:]
    s_err = layer.split_node(left_labels, right_labels, r, err=1e-4, is_last=carts == total_carts)
    split_errors.append(s_err)
#%%
import matplotlib.pyplot as plt
plt.plot(list(train_loss_dict.keys()), list(train_loss_dict.values()), label='Train RMSE')
plt.plot(list(test_loss_dict.keys()), list(test_loss_dict.values()), label='Test RMSE')
plt.xlabel('Number of Carriages')
plt.ylabel('RMSE')
plt.title('Train and Test RMSE vs Number of Carriages')
plt.legend()
#%%