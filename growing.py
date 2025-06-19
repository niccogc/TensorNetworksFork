#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
from tqdm.auto import tqdm
torch.set_default_dtype(torch.float64)
p = 1
degree = 5
def func(x):
    #return ((1+x)*(x**2)).sum(dim=-1, keepdim=True)
    x = x + 0.1*torch.randn_like(x) + 0.2
    return (100*(-1+x)*(-0.9+x)*(x)*(0.1+x)*(0.8+x)*(0.9+x)-2).sum(dim=-1, keepdim=True)
#coeffs = torch.randn(degree, 1, p)
# def func(x):
#     prod = ((x**3).sum(dim=-1, keepdim=True) + coeffs[0])
#     for i in range(degree-1):
#         prod = prod * ((x**3).sum(dim=-1, keepdim=True) + coeffs[i+1])*
#     return prod.sum(-1, keepdim=True)
# def func(x):
#     prod = (x + coeffs[0])
#     for i in range(degree-1):
#         prod = prod * (x + coeffs[i+1])
#     return prod.sum(dim=-1, keepdim=True)
x_train = torch.sort((torch.rand(10000, p)-0.5)*2).values
y_train = func(x_train).cuda()
x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).cuda()


x_test = torch.sort((torch.rand(1000, p)-0.5)*2).values
y_test = func(x_test).cuda()
x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).cuda()
#%%
N = 2
r = 8
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
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=-1, lr=1.0, eps=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=1, skip_second=True, direction='l2r', disable_tqdm=True)
#%%
for carts in tqdm(range(N+1, 7)):
    epss = np.geomspace(1.0, 1e-16, 6).tolist()
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

    layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=-1, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=3, skip_second=False, direction='r2l', disable_tqdm=True)
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
# %%
from matplotlib import pyplot as plt
# Plot y_pred vs y_true for the last cart
plt.figure(figsize=(10, 6))
x_sorted, indices = torch.sort(x_train[:, 0].squeeze().cpu())
y_sorted = y_train.squeeze().cpu()[indices]
x_pred = layer(x_train).squeeze().cpu()
y_pred_sorted = x_pred[indices]
plt.scatter(x_sorted, y_sorted, label='True Values', color='orange', s=10)
plt.scatter(x_sorted, y_pred_sorted, label='Predicted Values', color='red', s=10)
plt.xlabel('x')
#plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-10, 10)
#%%