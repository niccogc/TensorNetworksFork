#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
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

x_train, y_train, x_val, y_val, x_test, y_test = load_tabular_data('/work3/s183995/Nonlinear/data/cascaded_tanks_tensor.pt', device='cuda')

x_train = torch.tensor(x_train, device='cuda')
x_test = torch.tensor(x_test, device='cuda')
x_val = torch.tensor(x_val, device='cuda')

x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).to(dtype=torch.float64, device='cuda')
x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).to(dtype=torch.float64, device='cuda')
x_val = torch.cat((x_val, torch.ones((x_val.shape[0], 1), device=x_val.device)), dim=-1).to(dtype=torch.float64, device='cuda')

if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
y_train = y_train.to(dtype=torch.float64, device='cuda')
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)
y_test = y_test.to(dtype=torch.float64, device='cuda')
if y_val.ndim == 1:
    y_val = y_val.unsqueeze(1)
y_val = y_val.to(dtype=torch.float64, device='cuda')
#%%
import math
N = 30
r = 5
NUM_SWIPES = 10
method = 'ridge_cholesky'
epss = np.geomspace(1,  1e-8, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareBregFunction()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True).cuda()
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test - y_test)**2))
    print('Test RMSE:', rmse.item())
    return False

def D_reg_fn(node):
    size = math.prod(node.shape)
    return torch.eye(size, device=node.tensor.device, dtype=node.tensor.dtype) #+ torch.diag(torch.full((size-1,), 1, device=node.tensor.device, dtype=node.tensor.dtype), diagonal=1)

layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, D_reg_fn=D_reg_fn, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%
x = x_train
_, ind = torch.sort(x[:, 0])
y_pred = layer(x[ind])
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(y_pred.cpu().numpy(), label='True', alpha=0.5)
#%%