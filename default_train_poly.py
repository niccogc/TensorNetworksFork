#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
torch.set_default_dtype(torch.float64)
p = 1
def func(x):
    return (100*(-1+x)*(-0.9+x)*(x)*(0.1+x)*(0.8+x)*(0.9+x)-2).sum(dim=-1, keepdim=True)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
x_train = torch.sort((torch.rand(7, p)-0.5)*1.5, dim=0).values
y_train = func(x_train).cuda()
x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).cuda()


x_test = torch.sort((torch.rand(100, p)-0.5)*1.5, dim=0).values
y_test = func(x_test).cuda()
x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).cuda()

x_val = torch.sort((torch.rand(100, p)-0.5)*1.5, dim=0).values
y_val = func(x_val).cuda()
x_val = torch.cat((x_val, torch.ones((x_val.shape[0], 1), device=x_val.device)), dim=-1).cuda()

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
    
    # Plot all previous predictions with transparency
    for i, pred in enumerate(all_predictions):
        alpha = 0.3 + 0.7 * (i + 1) / len(all_predictions)  # More recent predictions are more opaque
        ax.plot(x_val[:,0].cpu().numpy(), pred[:,0], zorder=1, alpha=alpha)
    
    # Plot training data points on top
    ax.scatter(x_train[:,0].cpu().numpy(), y_train[:,0].cpu().numpy(), s=90, marker='*', color='blue', zorder=2)
    
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(f'Data - Training Progress (Step {len(all_predictions)})')
    ax.grid()
    plt.show()
    plt.close()  # Close the figure to free memory
#%%
N = 10
r = 8
NUM_SWIPES = 1
method = 'ridge_cholesky'
epss = [1e-14]*NUM_SWIPES*2# + np.geomspace(0.5, 0.01, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareBregFunction()
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True, perturb=True, seed=42).cuda()

train_loss_dict = {}
val_loss_dict = {}
def convergence_criterion():
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())
    plot_data(y_pred_val)
    return False

layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=True, direction='l2r', disable_tqdm=True)
convergence_criterion()
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
#%%