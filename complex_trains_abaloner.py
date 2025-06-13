#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from tensor.bregman import SquareComplexBregFunction, SquareBregFunction
from tensor.layers import ComplexTensorTrainLayer, TensorTrainLayer
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

x_train, y_train, x_val, y_val, x_test, y_test = load_tabular_data('/work3/s183995/Tabular/data/abaloner_tensor.pt', device='cuda')

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(x_train.cpu().numpy())
x_test = scaler.transform(x_test.cpu().numpy())
x_val = scaler.transform(x_val.cpu().numpy())
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
# Move everything to Complex128
x_train = x_train.to(dtype=torch.complex128)
x_test = x_test.to(dtype=torch.complex128)
x_val = x_val.to(dtype=torch.complex128)
y_train = y_train.to(dtype=torch.complex128)
y_test = y_test.to(dtype=torch.complex128)
y_val = y_val.to(dtype=torch.complex128)
#%%
N = 4
r = 100
NUM_SWIPES = 5
ORTHONORMALIZE = True
method = 'ridge_cholesky'
# Define Bregman function
bf = SquareComplexBregFunction()
layer = ComplexTensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=False).cuda()
if ORTHONORMALIZE:
    layer.tensor_network.orthonormalize_right()
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train.real - y_train.real)**2))
    print('Train RMSE:', rmse.item())
    #train_loss_dict[N] = rmse.item()
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test.real - y_test.real)**2))
    print('Test RMSE:', rmse.item())
    #test_loss_dict[N] = rmse.item()
    return False
epss = np.geomspace(0.5, 1e-6, NUM_SWIPES*2)
print(layer.num_parameters()*2)
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss.tolist(), convergence_criterion=convergence_criterion, orthonormalize=ORTHONORMALIZE, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)

print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
# %%
# Move everything to float64
x_train = x_train.to(dtype=torch.float64)
x_test = x_test.to(dtype=torch.float64)
x_val = x_val.to(dtype=torch.float64)
y_train = y_train.to(dtype=torch.float64)
y_test = y_test.to(dtype=torch.float64)
y_val = y_val.to(dtype=torch.float64)
# %%
import math
N = 4
r = int(100*math.sqrt(2))
print(r)
NUM_SWIPES = 5
ORTHONORMALIZE = True
method = 'ridge_cholesky'
# Define Bregman function
bf = SquareBregFunction()
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=False).cuda()
if ORTHONORMALIZE:
    layer.tensor_network.orthonormalize_right()
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train.real - y_train.real)**2))
    print('Train RMSE:', rmse.item())
    #train_loss_dict[N] = rmse.item()
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test.real - y_test.real)**2))
    print('Test RMSE:', rmse.item())
    #test_loss_dict[N] = rmse.item()
    return False
epss = np.geomspace(0.5, 1e-6, NUM_SWIPES*2)
print(layer.num_parameters())
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss.tolist(), convergence_criterion=convergence_criterion, orthonormalize=ORTHONORMALIZE, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)

print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
# %%