#%%
import torch
torch.set_default_dtype(torch.float64)

def poly_data(N, K, val_samples=1000, std=0.2):
    roots = std*torch.randn((N,), dtype=torch.float64) + 1
    X_train = std*torch.randn((K,), dtype=torch.float64) + 1
    X_val = torch.cat((torch.linspace(1-std, 1+std, val_samples//2, dtype=torch.float64),
                       std*torch.randn((val_samples//2,), dtype=torch.float64) + 1), dim=0).sort()[0]
    def func(x):
        return torch.prod((x.unsqueeze(-1) - roots), dim=-1, keepdim=True)
    y_train = func(X_train)
    y_val = func(X_val)
    return X_train, y_train, X_val, y_val, func, roots

X_train, y_train, X_val, y_val, func, roots = poly_data(6, 10)
#%%
import matplotlib.pyplot as plt
def plot_data(X, y, color='blue', fig=None, ax=None, s=4):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(X.cpu().numpy(), y.cpu().numpy(), color=color)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Data')
    ax.grid()
    return fig, ax
fig, ax = plot_data(X_val, y_val, color='orange')
fig, ax = plot_data(X_train, y_train, fig=fig, ax=ax, color='blue')
plt.show()
#%%
from sklearn.metrics import r2_score
import numpy as np
import scipy.special
def fit_poly(X_train, y_train, X_val, y_val, degree=3):
    feature_degrees = 1 + np.arange(degree).astype(int)
    X_train_poly = torch.tensor(scipy.special.eval_legendre(feature_degrees, X_train.unsqueeze(-1).cpu().numpy()), dtype=torch.float64)
    X_val_poly = torch.tensor(scipy.special.eval_legendre(feature_degrees, X_val.unsqueeze(-1).cpu().numpy()), dtype=torch.float64)
    coeffs = torch.linalg.lstsq(X_train_poly, y_train).solution

    y_train_pred = X_train_poly @ coeffs
    y_val_pred = X_val_poly @ coeffs

    train_score = r2_score(y_train.cpu().numpy(), y_train_pred.cpu().numpy())
    val_score = r2_score(y_val.cpu().numpy(), y_val_pred.cpu().numpy())
    print(f"Train R^2: {train_score:.4f}, Val R^2: {val_score:.4f}")
    return coeffs, y_train_pred, y_val_pred, train_score, val_score

coeffs, y_train_poly, y_val_poly, train_score, val_score = fit_poly(X_train, y_train, X_val, y_val, degree=10)
#%%
# Plot the polynomial fit
fig, ax = plt.subplots(figsize=(6, 6))
plot_data(X_val, y_val, color='blue', fig=fig, ax=ax)
plot_data(X_val, y_val_poly, color='orange', fig=fig, ax=ax)
# %%
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
fig, ax = plt.subplots(figsize=(14, 6))


def fit_tensor_train(X_train, y_train, X_val, y_val, degree=3, r=8, rel_err=1e-12, early_stopping=10):
    X_train = torch.stack((X_train, torch.ones_like(X_train)), dim=-1).cuda()
    X_val = torch.stack((X_val, torch.ones_like(X_val)), dim=-1).cuda()
    y_train = y_train.cuda()
    y_val = y_val.cuda()
    bf = SquareBregFunction()
    layer = TensorTrainLayer(degree, r, 2, output_shape=1, perturb=True).cuda()

    cur_degree = 1
    early_stop_count = 0
    best_degree = cur_degree
    best_r2_score = -np.inf
    best_train_r2_score = -np.inf
    best_state_dict = layer.node_states()
    def convergence_criterion():
        nonlocal cur_degree, early_stop_count, best_degree, best_r2_score, best_train_r2_score, best_state_dict
        y_pred_train = layer(X_train)
        train_score = r2_score(y_train.cpu().numpy(), y_pred_train.cpu().numpy())
        
        y_pred_val = layer(X_val)
        val_score = r2_score(y_val.cpu().numpy(), y_pred_val.cpu().numpy())

        plot_data(X_val[:, 0], y_pred_val, color=plt.get_cmap('viridis')(cur_degree/degree), fig=fig, ax=ax)

        # If the best_degree hasn't changed for `early_stopping` iterations, stop training
        if val_score > best_r2_score + rel_err:
            best_r2_score = val_score
            best_train_r2_score = train_score
            best_degree = cur_degree
            best_state_dict = layer.node_states()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stopping:
            print(f"Converged degree: {best_degree} with best R^2: {best_r2_score:.4f}")
            return True
        
        cur_degree += 1

        return False

    layer.tensor_network.accumulating_swipe(X_train, y_train, bf, convergence_criterion=convergence_criterion, eps=1e-14, method='ridge_cholesky', verbose=2, skip_second=True, disable_tqdm=True)
    layer.load_node_states(best_state_dict, set_value=True)
    
    y_val_pred = layer(X_val)
    y_train_pred = layer(X_train)

    train_score = r2_score(y_train.cpu().numpy(), y_train_pred.cpu().numpy())
    val_score = r2_score(y_val.cpu().numpy(), y_val_pred.cpu().numpy())
    print(f"Train R^2: {train_score:.4f}, Val R^2: {val_score:.4f}")
    return best_degree, layer, y_train_pred, y_val_pred, best_train_r2_score, best_r2_score

best_degree, layer, y_train_tt, y_val_tt, train_score, val_score = fit_tensor_train(X_train, y_train, X_val, y_val, degree=30, r=8)

plot_data(X_val, y_val, color='red', s=30, fig=fig, ax=ax)
#%%
# Plot the tensor train predictions
fig, ax = plt.subplots(figsize=(6, 6))
plot_data(X_val, y_val, color='blue', fig=fig, ax=ax)
#plot_data(X_val, y_val_poly, color='green', fig=fig, ax=ax)
plot_data(X_val, y_val_tt, color='orange', fig=fig, ax=ax)
# %%
