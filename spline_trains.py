#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorConvolutionTrainLayer, TensorConvOperatorLayer
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

quant_levels = 10
sigma = 1.0 / quant_levels

mus = torch.linspace(-1, 1, quant_levels, device=x_train.device).unsqueeze(0).unsqueeze(-1)
def transform(x, mus, sigma):
    return torch.exp(-(x.unsqueeze(1) - mus)**2 / (sigma)**2)
x_train = transform(x_train, mus, sigma)
x_test = transform(x_test, mus, sigma)
x_val = transform(x_val, mus, sigma)


x_train = torch.cat((x_train, torch.zeros((x_train.shape[0], 1, x_train.shape[2]), device=x_train.device)), dim=-2).cuda()
x_train = torch.cat((x_train, torch.zeros((x_train.shape[0], x_train.shape[1], 1), device=x_train.device)), dim=-1).cuda()
x_train[..., -1, -1] = 1.0

x_test = torch.cat((x_test, torch.zeros((x_test.shape[0], 1, x_test.shape[2]), device=x_test.device)), dim=-2).cuda()
x_test = torch.cat((x_test, torch.zeros((x_test.shape[0], x_test.shape[1], 1), device=x_test.device)), dim=-1).cuda()
x_test[..., -1, -1] = 1.0

x_val = torch.cat((x_val, torch.zeros((x_val.shape[0], 1, x_val.shape[2]), device=x_val.device)), dim=-2).cuda()
x_val = torch.cat((x_val, torch.zeros((x_val.shape[0], x_val.shape[1], 1), device=x_val.device)), dim=-1).cuda()
x_val[..., -1, -1] = 1.0

if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
y_train = y_train.to(dtype=torch.float64, device='cuda')
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)
y_test = y_test.to(dtype=torch.float64, device='cuda')
if y_val.ndim == 1:
    y_val = y_val.unsqueeze(1)
y_val = y_val.to(dtype=torch.float64, device='cuda')

#x_train = x_train.permute(0, 2, 1)
#x_test = x_test.permute(0, 2, 1)
#x_val = x_val.permute(0, 2, 1)
#%%
N = 10
r = 12
CB = -1
NUM_SWIPES = 5
method = 'ridge_cholesky'
epss = np.geomspace(1e-12, 1e-12, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareBregFunction()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=x_train.shape[1], patch_pixels=x_train.shape[2], output_shape=(1,), convolution_bond=CB, perturb=False).cuda()
#%%
train_loss_dict = {}
val_loss_dict = {}
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

layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.1, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=NUM_SWIPES == 1, direction='l2r', disable_tqdm=True)
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%
from matplotlib import pyplot as plt
plt.scatter(layer(x_train).cpu().detach().numpy(), y_train.cpu().detach().numpy(), s = 1)
# Plot "x"
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.scatter(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100), color='red', s=1)
plt.xlabel('Predicted')
plt.ylabel('True')
# %%
