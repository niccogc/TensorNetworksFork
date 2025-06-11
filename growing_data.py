#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainDMRGInfiLayer
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
from itertools import product
num_swipes_s = [5]
eps_min_s = [1e-4]
eps_max_s = [2.0]
err_s = [1e-2]
r_s = [4, 8, 12] # Added r_s for grid search
carts = 2
num_swipes = 1
eps_max = 1.0
eps_min = 1e-5
repeat = 0
# Initialize r, it will be set in the loop
r = r_s[0]
train_loss_dict = {}
val_loss_dict = {}
def convergence_criterion(_, __):
    global repeat, carts, num_swipes, eps_min, eps_max, err, r # Added r
    y_pred_train = layer(x_train)
    rmse = torch.sqrt(torch.mean((y_pred_train - y_train)**2))
    train_loss_dict[(repeat, carts, num_swipes, eps_min, eps_max, err, r)] = rmse.item() # Added r
    #print(carts, 'Train RMSE:', rmse.item())
    
    y_pred_val = layer(x_val)
    rmse = torch.sqrt(torch.mean((y_pred_val - y_val)**2))
    val_loss_dict[(repeat, carts, num_swipes, eps_min, eps_max, err, r)] = rmse.item() # Added r
    return False

method = 'ridge_cholesky'
for r, num_swipes, eps_min, eps_max, err in (t_bar:=tqdm(list(product(r_s, num_swipes_s, eps_min_s, eps_max_s, err_s)))): # Added r
    t_bar.set_description(f"R: {repeat} | r: {r} | NS: {num_swipes} | emin: {eps_min} | emax: {eps_max} | err: {err}") # Added r
    # Define Bregman function
    bf = SquareBregFunction()
    layer = TensorTrainDMRGInfiLayer(r, x_train.shape[1], output_shape=1, constrict_bond=True).cuda() # Use r from loop
    layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=256, lr=1.0, eps=eps_max, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=num_swipes, skip_second=False, direction='l2r', disable_tqdm=True)

    epss = np.geomspace(eps_max, eps_min, num_swipes+1)
    total_carts = 15
    
    for carts in range(2+1, total_carts+1):
        layer.grow_middle()
        for i in range(num_swipes):
            eps = epss[i]
            if not layer.tensor_network.accumulating_swipe(x_train, y_train, bf, batch_size=256, lr=1.0, eps=eps, convergence_criterion=convergence_criterion, orthonormalize=False, method=method, verbose=0, num_swipes=num_swipes, skip_second=False, direction='l2r', disable_tqdm=True):
                break
        node = layer.nodes[layer.num_carriages//2]
        left_labels = node.dim_labels[:2]
        right_labels = node.dim_labels[-2:]
        s_err = layer.split_node(left_labels, right_labels, r, err=err, is_last=carts == total_carts)
        t_bar.set_postfix_str(f"RMSE: {train_loss_dict[(repeat, carts, num_swipes, eps_min, eps_max, err, r)]:.4f} | {val_loss_dict[(repeat, carts, num_swipes, eps_min, eps_max, err, r)]:.4f}") # Added r
#%%
# Create pandas DataFrame
import pandas as pd
# Create a DataFrame from the dictionary
data_row = []
for (repeat, carts, num_swipes, eps_min, eps_max, err, r), rmse in train_loss_dict.items(): # Added r
    data_row.append({
        'repeat': repeat,
        'carts': carts,
        'num_swipes': num_swipes,
        'eps_min': eps_min,
        'eps_max': eps_max,
        'err': err,
        'r': r, # Added r
        'train_rmse': rmse
    })
for (repeat, carts, num_swipes, eps_min, eps_max, err, r), rmse in val_loss_dict.items(): # Added r
    data_row.append({
        'repeat': repeat,
        'carts': carts,
        'num_swipes': num_swipes,
        'eps_min': eps_min,
        'eps_max': eps_max,
        'err': err,
        'r': r, # Added r
        'val_rmse': rmse
    })
df = pd.DataFrame(data_row)
# Save the DataFrame to a CSV file
df.to_csv('/work3/s183995/Tabular/abaloner_growing.csv', index=False)
#%%
# Calculate mean and sem over repeats then plot RMSE as a function of each parameter
df_mean = df.groupby(['carts', 'num_swipes', 'eps_min', 'eps_max', 'err', 'r']).agg( # Added r
    train_rmse_mean=('train_rmse', 'mean'),
    train_rmse_sem=('train_rmse', lambda x: x.std() / np.sqrt(len(x))),
    val_rmse_mean=('val_rmse', 'mean'),
    val_rmse_sem=('val_rmse', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()
df_mean['carts'] = df_mean['carts'].astype(int)
df_mean['num_swipes'] = df_mean['num_swipes'].astype(int)
df_mean['eps_min'] = df_mean['eps_min'].astype(float)
df_mean['eps_max'] = df_mean['eps_max'].astype(float)
df_mean['err'] = df_mean['err'].astype(float)
df_mean['r'] = df_mean['r'].astype(int) # Added r
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set(style="whitegrid")
def plot_rmse(df, param, value_name, partition):

    plt.figure(figsize=(12, 7))
    # Use the new colormap API and sample colors evenly
    cmap = plt.colormaps['tab20']  # or try 'tab20b', 'tab20c'
    colors = [cmap(i / 20) for i in range(20)]
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 'D', '*', 'X', 's', '^', 'v', '<', '>', 'P']

    # Assign a unique (color, linestyle, marker) to each group
    style_combos = list(product(line_styles, markers, colors))
    group_keys = list(df.groupby(['num_swipes', 'eps_min', 'eps_max', 'err', 'r']).groups.keys()) # Added r

    for idx, key in enumerate(group_keys):
        grp = df[(df['num_swipes'] == key[0]) & (df['eps_min'] == key[1]) & (df['eps_max'] == key[2]) & (df['err'] == key[3]) & (df['r'] == key[4])] # Added r
        ls, marker, color = style_combos[idx]
        plt.plot(
            grp[param],
            grp[partition+"_"+value_name],
            label=f'NS: {key[0]}, emin: {key[1]}, emax: {key[2]}, err: {key[3]}, r: {key[4]}', # Added r
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            alpha=0.85
        )
    plt.xlabel(param)
    plt.ylabel(f'{partition} RMSE')
    plt.title(f'{partition} RMSE vs {param}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()
# Only
df_filter = df_mean#[(df_mean['num_swipes'] == 6) & (df_mean['r'] == 12)]
plot_rmse(df_filter, 'carts', 'rmse_mean', 'val')
#%%
# Find argmin parameters
def find_best_params(df, partition):
    best_params = df.loc[df[f'{partition}_rmse_mean'].idxmin()]
    return best_params
best_params_train = find_best_params(df_mean, 'train')
best_params_val = find_best_params(df_mean, 'val')
print("Best parameters for training RMSE:")
print(best_params_train)
print("Best parameters for validation RMSE:")
print(best_params_val)
# %%
