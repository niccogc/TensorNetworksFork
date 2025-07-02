#%%
import os
os.environ["cpu_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from tensor.data_compression import train_concat
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
torch.set_default_dtype(torch.float64)
p = 1
degree = 5
S= 5000
def func(x):
    #return ((1+x)*(x**2)).sum(dim=-1, keepdim=True)
    #x = x + 0.2
    # *(x-0.2) * (x+0.8) *(x+0.9) * (x-0.3) * (x-0.1) * (x+0.33) * (x-1.2)
    f = (np.add(2.0 * x, np.cos(x * 5))).sum(dim=-1, keepdim=True)#(0.2 + (x-0.5) * (x+0.1) * x).sum(dim=-1, keepdim=True) #(100*(-1+x)*(-0.9+x)*(x)*(0.1+x)*(0.8+x)*(0.9+x)-2).sum(dim=-1, keepdim=True)
    return f + torch.randn_like(f, device=f.device) * 0.3

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
torch.manual_seed(41)

x_train = torch.sort((torch.rand(S, p)-0.5)*2, dim=0).values
y_train = func(x_train).cpu()
x_train = torch.cat((x_train, torch.ones((x_train.shape[0], 1), device=x_train.device)), dim=-1).cpu()


x_test = torch.sort((torch.rand(S, p)-0.5)*1.5, dim=0).values
y_test = func(x_test).cpu()
x_test = torch.cat((x_test, torch.ones((x_test.shape[0], 1), device=x_test.device)), dim=-1).cpu()

x_val = torch.sort((torch.rand(S, p)-0.5)*1.5, dim=0).values
y_val = func(x_val).cpu()
x_val = torch.cat((x_val, torch.ones((x_val.shape[0], 1), device=x_val.device)), dim=-1).cpu()

if y_train.ndim == 1:
    y_train = y_train.unsqueeze(1)
y_train = y_train.to(dtype=torch.float64, device='cpu')
if y_test.ndim == 1:
    y_test = y_test.unsqueeze(1)
y_test = y_test.to(dtype=torch.float64, device='cpu')
if y_val.ndim == 1:
    y_val = y_val.unsqueeze(1)
y_val = y_val.to(dtype=torch.float64, device='cpu')
#%%
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
N = 6
r = 3
NUM_SWIPES = 1
method = 'ridge_cholesky'
epss = [1e-12]*NUM_SWIPES*2# + np.geomspace(0.5, 0.01, NUM_SWIPES*2).tolist()
# Define Bregman function
bf = SquareBregFunction()
layer = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True, perturb=True, seed=42).cpu()

layer1 = TensorTrainLayer(N, r, x_train.shape[1], output_shape=1, constrict_bond=True, perturb=False, seed=46).cpu()

train_loss_dict = {}
val_loss_dict = {}
def convergence_criterion():
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    print('Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    print('Val RMSE:', rmse.item())
    # plot_data(y_pred_val)
    return False

layer.tensor_network.accumulating_swipe(x_train[:2500], y_train[2500:], bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)

layer1.tensor_network.accumulating_swipe(x_train[2500:], y_train[2500:], bf, batch_size=512, lr=1.0, eps=epss, eps_r=0.5, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=2, num_swipes=NUM_SWIPES, skip_second=False, direction='l2r', disable_tqdm=True)

convergence_criterion()
list1=[i.tensor.unsqueeze(1) for i in layer.nodes]
list2=[i.tensor.unsqueeze(1) for i in layer1.nodes]
list1[-1] = list1[-1].unsqueeze(-1)
list2[-1] = list2[-1].unsqueeze(-1)
mean= train_concat(list1, list2)
mean=[i.squeeze() for i in mean]
mean[0] = mean[0].unsqueeze(0)
mean[-1] = mean[-1].unsqueeze(-1)/2
for i in mean:
    print(i.shape)
layermean = TensorTrainLayer(N, r*2, x_train.shape[1], nodes=mean)
print("t1", layer.nodes[1].tensor[:,0,:], )
print("t2", layer1.nodes[1].tensor[:,0,:])
print("tm", layermean.nodes[1].tensor[:,0,:])
l1=layer(x_train)
l2=layer1(x_train)
lm=layermean(x_train)

print(l1.shape, l2.shape, lm.shape)
print("MEan",lm, "l1", l1, "l2", l2, "lm", (l1+l2)/2)
print("Train:",(y_train.real - layer(x_train).real).square().mean().sqrt())
print("Val:",(y_val.real - layer(x_val).real).square().mean().sqrt())
print("Train1:",(y_train.real - layer1(x_train).real).square().mean().sqrt())
print("Val1:",(y_val.real - layer1(x_val).real).square().mean().sqrt())
print("Trainmean:",(y_train.real - layermean(x_train).unsqueeze(-1)).square().mean().sqrt())
print("Valmean:",(y_val.real - layermean(x_val).unsqueeze(-1)).square().mean().sqrt())

#%%
