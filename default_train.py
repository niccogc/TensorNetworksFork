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
N = 5
r = 12
NUM_SWIPES = 5
method = 'ridge_cholesky'
epss = np.geomspace(0.02206915991899018, 0.000317837364967951, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareBregFunction()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True).cuda()
train_loss_dict = {}
val_loss_dict = {}
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())
    return False

layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)
convergence_criterion(None, None)
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%
y_train_pred = layer(x_train).cpu().numpy()
y_val_pred = layer(x_val).cpu().numpy()
y_test_pred = layer(x_test).cpu().numpy()
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_train.cpu().numpy(), y_train_pred, label='Train', alpha=0.5)
plt.scatter(y_val.cpu().numpy(), y_val_pred, label='Validation', alpha=0.5)
plt.scatter(y_test.cpu().numpy(), y_test_pred, label='Test', alpha=0.5)
plt.plot([y_train.amin().item(), y_train.amax().item()], [y_train.amin().item(), y_train.amax().item()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Ridge Regression Predictions')
plt.legend()
plt.show()
#%%
# Convert to pandas DataFrame
import pandas as pd
train_df = pd.DataFrame(x_train.cpu().numpy(), columns=[f'feature_{i}' for i in range(x_train.shape[1])])
train_df['target'] = y_train.cpu().numpy()
train_df['split'] = 'train'

val_df = pd.DataFrame(x_val.cpu().numpy(), columns=[f'feature_{i}' for i in range(x_val.shape[1])])
val_df['target'] = y_val.cpu().numpy()
val_df['split'] = 'val'

test_df = pd.DataFrame(x_test.cpu().numpy(), columns=[f'feature_{i}' for i in range(x_test.shape[1])])
test_df['target'] = y_test.cpu().numpy()
test_df['split'] = 'test'

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
# Visualize as a grid scatter plot using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.pairplot(all_df, diag_kind='kde', markers='o', plot_kws={'alpha': 0.5}, hue='split')
# %%
