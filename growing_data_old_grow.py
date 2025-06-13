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
N = 2
r = 10
method = 'ridge_cholesky'
# Define Bregman function
bf = SquareBregFunction()
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=False).cuda()
y_pred_prev = layer(x_train)
train_loss_dict = {}
test_loss_dict = {}
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    #print('Train RMSE:', rmse.item())
    train_loss_dict[N] = rmse.item()
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test - y_test)**2))
    #print('Test RMSE:', rmse.item())
    test_loss_dict[N] = rmse.item()
    return False
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=1, skip_second=True, direction='l2r', disable_tqdm=True)
#%%
NUM_SWIPES = 5
for carts in tqdm(range(N+1, 11)):
    epss = np.geomspace(5.0, 1e-2, 2*NUM_SWIPES).tolist()
    layer.grow_cart().cuda()

    def convergence_criterion(_, __):
        y_pred_train = layer(x_train)
        rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
        #print('Train RMSE:', rmse.item())
        train_loss_dict[carts] = rmse.item()
        
        y_pred_test = layer(x_test)
        rmse = torch.sqrt(torch.mean((y_pred_test - y_test)**2))
        #print('Test RMSE:', rmse.item())
        test_loss_dict[carts] = rmse.item()

    layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='r2l', disable_tqdm=True)
epss = np.geomspace(1.0, 1e-3, 2*NUM_SWIPES).tolist()
carts = 11
def convergence_criterion(_, __):
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    #print('Train RMSE:', rmse.item())
    train_loss_dict[carts] = rmse.item()
    
    y_pred_test = layer(x_test)
    rmse = torch.sqrt(torch.mean((y_pred_test - y_test)**2))
    #print('Test RMSE:', rmse.item())
    test_loss_dict[carts] = rmse.item()
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=5, skip_second=False, direction='r2l', disable_tqdm=True)
#%%
# Print minimum test loss
min_test_loss = min(test_loss_dict.values())
min_test_carts = min(test_loss_dict, key=test_loss_dict.get)
print(f'Minimum test loss: {min_test_loss} at {min_test_carts} carts')
#%%
# Plot the results
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
# Plot carts
x_carts = list(train_loss_dict.keys())#[-10:]
y_carts_train = list(train_loss_dict.values())#[-10:]
y_carts_test = list(test_loss_dict.values())#[-10:]
plt.plot(x_carts, y_carts_train, label='Train RMSE', marker='o')
plt.plot(x_carts, y_carts_test, label='Test RMSE', marker='o')
#plt.ylim(0, 2)
plt.xlabel('Number of Carts')
plt.ylabel('RMSE')
plt.title('Train and Test RMSE vs Number of Carts')
plt.legend()
#%%
y_carts_train_d = np.array(y_carts_train[:-1]) - np.array(y_carts_train[1:])
y_carts_test_d = np.array(y_carts_test[:-1]) - np.array(y_carts_test[1:])
plt.figure(figsize=(10, 6))
plt.plot(x_carts[:-1], y_carts_train_d, label='Train RMSE', marker='o')
plt.plot(x_carts[:-1], y_carts_test_d, label='Test RMSE', marker='o')
plt.xlabel('Number of Carts')
plt.ylabel('RMSE Difference')
# %%