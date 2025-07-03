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
import matplotlib.pyplot as plt

# Store all predictions for cumulative plotting
all_predictions = []

def plot_data(y_pred):
    # Add new prediction to our collection
    all_predictions.append(y_pred.cpu().numpy())
    
    # Create a fresh figure with all data
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot all previous predictions with color progression
    for i, pred in enumerate(all_predictions):
        # Color progression from early (red/orange) to recent (blue/purple)
        color_ratio = i / max(1, len(all_predictions) - 1)  # Avoid division by zero
        color = plt.cm.plasma(color_ratio)  # Use plasma colormap for nice progression
        
        # Alpha progression - more recent predictions are more opaque
        alpha = 0.3 + 0.7 * (i + 1) / len(all_predictions)
        
        ax.scatter(x_val[:,0].cpu().numpy(), pred[:,0], 
                  color=color, alpha=alpha, s=20, marker='o', 
                  zorder=2, label=f'Step {i+1}' if i == len(all_predictions)-1 else None)
    
    ax.scatter(x_train[:,0].cpu().numpy(), y_train[:,0].cpu().numpy(), 
              s=90, alpha=0.5, marker='*', color='black', zorder=1, label='Training Data')
    
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(f'Data - Training Progress (Step {len(all_predictions)})')
    ax.grid()
    ax.legend()
    plt.show()
    plt.close()  # Close the figure to free memory
#%%
from tensor.bregman import UncertaintyAutogradLoss
N = 4
r = 20
NUM_SWIPES = 10
method = 'ridge_cholesky'
epss = np.geomspace(10, 1e-10, 2*NUM_SWIPES).tolist()
# Define Bregman function
bf = UncertaintyAutogradLoss()#SquareBregFunction()
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=(2,), constrict_bond=True, perturb=False, seed=42).cuda()
#%%
train_loss_dict = {}
val_loss_dict = {}
def convergence_criterion():
    y_pred_train = layer(x_train)[..., 0]
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)[..., 0]
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())

    r2 = 1 - torch.sum((y_pred_val - y_val)**2) / torch.sum((y_val - y_val.mean())**2)
    print('Val R2:', r2.item())
    #plot_data(y_pred_val)
    return False
#for num_swipes in range(NUM_SWIPES):
    #layer.tensor_network.train_nodes[-num_swipes-1:(-num_swipes) if num_swipes > 0 else None]
    #torch.nn.init.trunc_normal_(layer.tensor_network.train_nodes[num_swipes].tensor, mean=0.0, std=0.02, a=-0.04, b=0.04)
    #layer.tensor_network.accumulating_swipe(x_train, y_train, bf, node_order=layer.tensor_network.train_nodes[num_swipes:num_swipes+1], batch_size=512, lr=1.0, eps=epss[num_swipes], eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=1, skip_second=True, direction='l2r', disable_tqdm=True)
layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=True, direction='l2r', disable_tqdm=True)
convergence_criterion()
print("Train:",(y_train.real - layer(x_train)[..., 0].real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val)[..., 0].real).square().mean().sqrt())
#%%
from matplotlib import pyplot as plt
plt.scatter(layer(x_val)[..., 0].cpu().numpy(), y_val.cpu().numpy(), s=1, alpha=0.5, marker='o', label='Predictions')
plt.xlabel('Predicted Output')
plt.ylabel('True Output')
plt.title('Predicted vs True Output')
# %%
