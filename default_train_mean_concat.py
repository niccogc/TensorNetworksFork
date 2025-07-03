#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

x_train, y_train, x_val, y_val, x_test, y_test = load_tabular_data('/work3/s183995/Tabular/data/processed/house_tensor.pt', device='cuda')

x_train = torch.tensor(x_train, device='cuda')
x_std, x_mean = torch.std_mean(x_train, dim=0, unbiased=False, keepdim=True)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
x_val = (x_val - x_mean) / x_std

x_min, x_max = x_train.amin(dim=0, keepdim=True), x_train.amax(dim=0, keepdim=True)
x_test = torch.tensor(x_test, device='cuda')
x_val = torch.tensor(x_val, device='cuda')

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
SPLITS = 5
# Divide the training data into splits
x_train_splits = []
y_train_splits = []
split_size = x_train.shape[0] // SPLITS
for i in range(SPLITS):
    start = i * split_size
    end = (i + 1) * split_size if i < SPLITS - 1 else x_train.shape[0]
    x_train_splits.append(x_train[start:end])
    y_train_splits.append(y_train[start:end])
#%%
N = 10
r = 2
NUM_SWIPES = 1
method = 'ridge_cholesky'
epss = np.geomspace(1e-12, 1e-10, 2*NUM_SWIPES).tolist()
# Define Bregman function
bf = SquareBregFunction()
layers = []
for i in range(SPLITS):
    layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True, perturb=True, seed=42).cuda()
    layers.append(layer)
#%%
from tensor.layers import concatenate_trains
train_loss_dict = {}
val_loss_dict = {}
for i, layer in enumerate(layers):
    def convergence_criterion():
        y_pred_train = layer(x_train_splits[i])
        rmse = torch.sqrt(torch.mean((y_pred_train - y_train_splits[i])**2))
        print(i, 'Train RMSE:', rmse.item())

        y_pred_val = layer(x_val)
        rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
        print(i, 'Val RMSE:', rmse.item())

        r2 = 1 - torch.sum((y_pred_val - y_val)**2) / torch.sum((y_val - y_val.mean())**2)
        print(i, 'Val R2:', r2.item())
        return False
    layer.tensor_network.accumulating_swipe(x_train_splits[i], y_train_splits[i], bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=NUM_SWIPES, skip_second=True, direction='l2r', disable_tqdm=True)
layer = concatenate_trains(layers).cuda()
def convergence_criterion():
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())

    r2 = 1 - torch.sum((y_pred_val - y_val)**2) / torch.sum((y_val - y_val.mean())**2)
    print('Val R2:', r2.item())
    return False
convergence_criterion()
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%