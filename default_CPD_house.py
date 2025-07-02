#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensor.bregman import AutogradLoss
from tqdm import tqdm
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
if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
y_train = y_train.to(dtype=torch.float64, device='cuda')
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)
y_test = y_test.to(dtype=torch.float64, device='cuda')
if y_val.ndim == 1:
    y_val = y_val.unsqueeze(1)
y_val = y_val.to(dtype=torch.float64, device='cuda')

x_train_ = torch.tensor(x_train, device='cuda')
x_test_ = torch.tensor(x_test, device='cuda')
x_val_ = torch.tensor(x_val, device='cuda')
x_std, x_mean = torch.std_mean(x_train_, dim=0, unbiased=False, keepdim=True)
train_loss_dict = {}
val_loss_dict = {}
val_r2_dict = {}
for std in tqdm([1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1.0, 2.0, 3.0, 4.0, 10, 1e2, 1e3, 1e4, 1e6]):
    x_train = (x_train_ - x_mean+1) / (std*x_std)
    x_test = (x_test_ - x_mean+1) / (std*x_std)
    x_val = (x_val_ - x_mean+1) / (std*x_std)

    print(x_train.mean(), x_train.std(), x_train.amin(), x_train.amax())

    x_min, x_max = x_train.amin(dim=0, keepdim=True), x_train.amax(dim=0, keepdim=True)
    x_test = torch.tensor(x_test, device='cuda')
    x_val = torch.tensor(x_val, device='cuda')

    eps_val = 1e-2

    x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).to(dtype=torch.float64, device='cuda')
    x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).to(dtype=torch.float64, device='cuda')
    x_val = torch.cat((x_val, torch.ones((x_val.shape[0], 1), device=x_val.device)), dim=-1).to(dtype=torch.float64, device='cuda')


    from tensor.layers import CPDLayer
    N = 5
    r = 10
    NUM_SWIPES = 5
    for N in range(8,9):
        for r in range(11, 16):
            method = 'ridge_cholesky'
            epss = np.geomspace(1e-12, 1e-10, 2*NUM_SWIPES).tolist()
            # Define Bregman function
            loss_fn = AutogradLoss(torch.nn.MSELoss(reduction='none'))
            layer = CPDLayer(N, r, x_train.shape[-1], output_shape=(1,)).cuda()
            def convergence_criterion():
                y_pred_train = layer(x_train)
                rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
                print('Train RMSE:', rmse.item())
                train_loss_dict[(N, r, std)] = rmse.item()
                
                y_pred_val = layer(x_val)
                rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
                print('Val RMSE:', rmse.item())
                val_loss_dict[(N, r, std)] = rmse.item()

                r2 = 1 - torch.sum((y_pred_val - y_val)**2) / torch.sum((y_val - y_val.mean())**2)
                print('Val R2:', r2.item())
                val_r2_dict[(N, r, std)] = r2.item()
                return False
            layer.tensor_network.accumulating_swipe(x_train, y_train, loss_fn, batch_size=-1, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)
            convergence_criterion()
#%%
# Print min over rmse and the corresponding N, r
min_rmse = min(val_loss_dict.values())
best_params = [k for k, v in val_loss_dict.items() if v == min_rmse]
print(f"Best RMSE: {min_rmse} for parameters: {best_params}")
# Print max over r2 and the corresponding N, r
max_r2 = max(val_r2_dict.values())
best_r2_params = [k for k, v in val_r2_dict.items() if v == max_r2]
print(f"Best R2: {max_r2} for parameters: {best_r2_params}")
# %%