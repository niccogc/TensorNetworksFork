#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import math
from tensor.bregman import SquareComplexBregFunction
from tensor.layers import TensorTrainLayer
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
torch.set_default_dtype(torch.float64)

def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    x_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    x_val = data['X_val'].to(device)
    y_val = data['y_val'].to(device)
    x_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = load_tabular_data('/work3/s183995/Tabular/data/processed/house_tensor.pt', device='cuda')

x_train = x_train.to(dtype=torch.float64, device='cuda')
x_test = x_test.to(dtype=torch.float64, device='cuda')
x_val = x_val.to(dtype=torch.float64, device='cuda')

#x_train = torch.tensor(x_train, device='cuda')
x_std, x_mean = torch.std_mean(x_train, dim=0, unbiased=False, keepdim=True)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
x_val = (x_val - x_mean) / x_std

x_min, x_max = x_train.amin(dim=0, keepdim=True), x_train.amax(dim=0, keepdim=True)
#x_test = torch.tensor(x_test, device='cuda')
#x_val = torch.tensor(x_val, device='cuda')

eps_val = 1e-2

x_test_mask = ((x_min <= x_test) & (x_test <= x_max)).all(-1)
x_val_mask = ((x_min <= x_val) & (x_val <= x_max)).all(-1)

# Mask out
# print(x_test.shape, x_val.shape)
# x_test = x_test[x_test_mask]
# x_val = x_val[x_val_mask]
# y_test = y_test[x_test_mask]
# y_val = y_val[x_val_mask]
# print(x_test.shape, x_val.shape)

# Clamp to min/max
x_test = torch.clamp(x_test, x_min, x_max)
x_val = torch.clamp(x_val, x_min, x_max)

# x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).to(dtype=torch.float64, device='cuda')
# x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).to(dtype=torch.float64, device='cuda')
# x_val = torch.cat((x_val, torch.ones((x_val.shape[0], 1), device=x_val.device)), dim=-1).to(dtype=torch.float64, device='cuda')

if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
y_train = y_train.to(dtype=torch.float64, device='cuda')
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)
y_test = y_test.to(dtype=torch.float64, device='cuda')
if y_val.ndim == 1:
    y_val = y_val.unsqueeze(1)
y_val = y_val.to(dtype=torch.float64, device='cuda')

def dense_fourier_basis(x, num_freqs):
    N = num_freqs
    n = torch.cat((torch.arange(-num_freqs//2, 0, device=x.device), torch.arange(1, num_freqs//2, device=x.device), torch.tensor([0], device=x.device)))[None, :]
    basis = torch.exp(-2j * math.pi * x[..., None] * n / N )
    return [basis[:, i] for i in range(basis.shape[1])]

def sparse_fourier_basis(x, freqs, period=1.0):
    return [torch.exp(-2j * math.pi * f * x / period) for f in freqs]

def get_powers_of_two(n):
    powers = list(reversed((-2.0**(np.arange(n))).tolist())) + [0] + (2.0**(np.arange(n))).tolist()
    period = max([abs(p) for p in powers]) * 2
    return powers, period

num_freqs = 16
dense = False

if dense:
    x_train = dense_fourier_basis(x_train, num_freqs=num_freqs)
    x_val = dense_fourier_basis(x_val, num_freqs=num_freqs)
    x_test = dense_fourier_basis(x_test, num_freqs=num_freqs)
else:
    powers, period = get_powers_of_two(num_freqs)
    period = period
    powers = np.linspace(0, 15, num_freqs)#[3,5,6,7] + powers
    x_train = sparse_fourier_basis(x_train, freqs=powers, period=period)
    x_val = sparse_fourier_basis(x_val, freqs=powers, period=period)
    x_test = sparse_fourier_basis(x_test, freqs=powers, period=period)
#%%
N = len(x_train)
r = 4
NUM_SWIPES = 4
batch_size = 8192
method = 'ridge_cholesky'
epss = 1e-13#np.geomspace(5, 1e-6, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareComplexBregFunction()
layer = TensorTrainLayer(N, r, x_train[0].shape[1], output_shape=1, constrict_bond=True, perturb=False, seed=42, dtype=torch.complex128).cuda()
#%%
from collections import defaultdict
train_loss_dict = defaultdict(list)
val_loss_dict = defaultdict(list)
def convergence_criterion():
    y_pred_train = layer(x_train).real
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())

    y_pred_val = layer(x_val).real
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())

    r2 = 1 - torch.sum((y_pred_val - y_val)**2) / torch.sum((y_val - y_val.mean())**2)
    print('Val R2:', r2.item())
    val_loss_dict['rmse'].append(rmse.item())
    val_loss_dict['r2'].append(r2.item())
    #plot_data(y_pred_val)
    return False
# Print max r2 and max rmse
# for i in range(N):
#     layer.tensor_network.accumulating_swipe(x_train, y_train, bf, node_order=[layer.tensor_network.train_nodes[i]], batch_size=512, lr=1.0, eps=epss[i], eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=1, skip_second=True, direction='l2r', disable_tqdm=True)
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=batch_size, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)
convergence_criterion()
print('Max R2:', max(val_loss_dict['r2']))
print('Min RMSE:', min(val_loss_dict['rmse']))
#%%