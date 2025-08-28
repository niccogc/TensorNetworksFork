#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

from tensor.bregman import SquareBregFunction, AutogradLoss
from tensor.module import TensorTrainRegressor, EarlyStopping
from tensor.layers import CPDLayer, TensorTrainLayer
from scipy import special
from sklearn.preprocessing import PolynomialFeatures
import math
import time

from torchvision import datasets
from sklearn.preprocessing import QuantileTransformer

# Download and load entire MNIST dataset
train_dataset = datasets.MNIST(
    root="/work3/aveno/MNIST",
    train=True,
    download=True,
)

test_dataset = datasets.MNIST(
    root="/work3/aveno/MNIST",
    train=False,
    download=True,
)

# Access the data and targets directly as tensors
X_train, y_train = train_dataset.data, train_dataset.targets
X_test, y_test = test_dataset.data, test_dataset.targets

# Flatten X and normalize using QuantileTransformer(output_distribution="uniform")
X_train = X_train.view(X_train.size(0), -1).numpy() / 255.0
X_test = X_test.view(X_test.size(0), -1).numpy() / 255.0

X_quant = QuantileTransformer(output_distribution="uniform")
X_train = X_quant.fit_transform(X_train)
X_test = X_quant.transform(X_test)

# One-hot encode y
y_train = np.eye(10)[y_train.numpy()]
y_test = np.eye(10)[y_test.numpy()]
#%%
tt = TensorTrainRegressor(
    num_swipes=20,
    eps_start=1e-12,
    eps_end=1e-12,
    N=5,
    r=8,
    linear_dim=16,
    output_dim=10,
    batch_size=2048,
    constrict_bond=False,
    perturb=False,
    seed=42,
    device='cuda',
    bf=AutogradLoss(torch.nn.MSELoss(reduction='none')),
    lr=1.0,
    method="ridge_cholesky",
    model_type="tt_type1_bias_first_no_train_linear", #tt_type1_bias_first_no_train_linear
    verbose=1
)
tt.fit(X_train, y_train)
# evaluate on the test set
y_pred_test = tt.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred_test, axis=1))
print("R2 test:", r2_test, "RMSE test:", rmse_test, "Accuracy:", accuracy)
#%%

def fit_poly_mononomial(X, y, degree, include_bias=True):
    """
    X_train: Numpy array of shape (B, d)
    y: Numpy array of shape (B,1)
    degree: total degree (int)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    Phi = poly.fit_transform(X)          # (B, n_terms)
    y = y.reshape(-1, 1)

    Phi = torch.tensor(Phi, dtype=torch.float64, device='cuda')
    y = torch.tensor(y, dtype=torch.float64, device='cuda')
    lstsq = torch.linalg.lstsq(Phi, y)
    coeffs = lstsq.solution  # (n_terms, 1)
    rank = lstsq.rank
    
    coeffs = coeffs.cpu().numpy()  # convert to numpy array

    return poly, coeffs, rank

def evaluate_poly(X, coeffs, poly):
    """
    Evaluate the polynomial with coefficients coeffs at X.

    X: Numpy array of shape (B, d)
    coeffs: Numpy array of shape (n_terms, 1)
    poly: PolynomialFeatures instance used to generate coeffs
    """
    Phi = poly.transform(X)          # (B, n_terms)
    y_pred = Phi.dot(coeffs)         # (B, 1)
    return y_pred

def evaluate_polynomial_regression(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    d,
    abs_err=0.0,
    rel_err=0.0,
    early_stopping=5,
    verbose=0
):
    current = {'poly': None, 'coeffs': None, 'rank': None}

    def model_predict(X):
        return evaluate_poly(X, current['coeffs'], current['poly'])

    def get_weights():
        return (current['poly'], current['coeffs'], current['rank'])

    loss_fn = root_mean_squared_error

    es = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=model_predict,
        get_model_weights=get_weights,
        loss_fn=loss_fn,
        abs_err=abs_err,
        rel_err=rel_err,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=1
    )

    for deg in range(1, max_degree + 1):
        if special.comb(d + deg, d) >= 12_000:
            break

        poly, coeffs, rank = fit_poly_mononomial(
            X_train, y_train, degree=deg, include_bias=True
        )

        current['poly'], current['coeffs'], current['rank'] = poly, coeffs, rank

        if es.convergence_criterion():
            break
    
    best_summary = es.best_summary()
    best_poly, best_coeffs, best_rank = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']
    val_history = es.val_history
    time_history = es.time_history
    if best_poly is None:
        return np.nan, np.nan, None, None, np.nan, None, None, None

    y_test_pred = evaluate_poly(X_test, best_coeffs, best_poly)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    num_params = np.prod(best_coeffs.shape)

    return test_r2, test_rmse, best_poly, best_coeffs, best_degree, num_params, best_rank, val_history, time_history

def prepare_torch_data(
    X, y,
    X_val=None, y_val=None,
    validation_split=0.2,
    split_train=False,
    seed=42,
    device='cuda'
):
    # Convert to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float64, device=device)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # Validation split if not provided
    if X_val is None or y_val is None:
        if split_train:
            n = X.shape[0]
            idx = np.arange(n)
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            split = int(n * (1 - validation_split))
            train_idx, val_idx = idx[:split], idx[split:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y
    else:
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float64, device=device)
        if isinstance(y_val, np.ndarray):
            y_val = torch.tensor(y_val, dtype=torch.float64, device=device)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(1)
        X_train, y_train = X, y

    # Append 1 to X for bias term
    X_train = torch.cat((X_train, torch.ones((X_train.shape[0], 1), dtype=torch.float64, device=device)), dim=1)
    X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), dtype=torch.float64, device=device)), dim=1)

    return X_train, y_train, X_val, y_val


def cpd_predict(cpd, X, device='cuda'):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    # Append 1 to X for bias term
    X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)), dim=1)
    y_pred = cpd(X)
    return y_pred.detach().cpu().numpy()

def evaluate_cpd(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    rank,
    num_swipes=5,
    eps_start=1e-12,
    eps_end=1e-10,
    early_stopping=5,
    split_train=False,
    method='ridge_cholesky',
    timeout=10,
    random_state=42,
    verbose=0
):
    # Set torch seed
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    X_train, y_train, X_val, y_val = prepare_torch_data(
        X_train, y_train,
        X_val, y_val,
        split_train=split_train,
        seed=random_state,
        device='cuda'
    )

    early_stopping = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=lambda X: None, #cpd,
        get_model_weights=lambda: None, #lambda: cpd.node_states(),
        loss_fn=root_mean_squared_error_torch,
        abs_err=1e-6,
        rel_err=1e-4,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=2
    )

    loss_fn = AutogradLoss(torch.nn.MSELoss(reduction='none'))

    time_start = time.time()

    for deg in range(2, max_degree + 1):
        cpd = CPDLayer(
            num_factors=deg,
            rank=rank,
            input_features=X_train.shape[1],
            output_shape=1,
        ).cuda()
        early_stopping.model_predict = cpd
        early_stopping.get_model_weights = lambda: cpd.node_states()
        converged = cpd.tensor_network.accumulating_swipe(
            X_train, y_train, loss_fn,
            batch_size=-1,
            convergence_criterion=None,
            eps=np.geomspace(eps_start, eps_end, num=num_swipes*2).tolist(),
            method=method,
            skip_second=False,
            lr=1.0,
            orthonormalize=False,
            verbose=verbose == 1,
            num_swipes=num_swipes,
            direction='l2r',
            disable_tqdm=True
        )

        if early_stopping.convergence_criterion() or (time.time() - time_start) > timeout:
            break
    
    best_summary = early_stopping.best_summary()
    best_state_dict = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']

    singular = not converged

    # Restore best state if available
    cpd = CPDLayer(
        num_factors=best_degree,
        rank=rank,
        input_features=X_train.shape[1],
        output_shape=1,
    ).cuda()
    if best_state_dict is not None:
        cpd.load_node_states(best_state_dict, set_value=True)
    y_pred_test = cpd_predict(cpd, X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    num_params = sum(p.tensor.numel() for i, p in enumerate(cpd.tensor_network.train_nodes))
    val_history = early_stopping.val_history
    time_history = early_stopping.time_history
    return r2_test, rmse_test, singular, cpd, best_degree, num_params, val_history, time_history

def tt_full_predict(tt_model, X, device='cuda'):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    # Append 1 to X for bias term
    X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)), dim=1)
    y_pred = tt_model(X)
    return y_pred.detach().cpu().numpy()

def evaluate_tt_full(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree,
    rank,
    num_swipes=5,
    eps_start=1e-12,
    eps_end=1e-10,
    early_stopping=5,
    split_train=False,
    method='ridge_cholesky',
    timeout=10,
    random_state=42,
    verbose=0
):
    # Set torch seed
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    X_train, y_train, X_val, y_val = prepare_torch_data(
        X_train, y_train,
        X_val, y_val,
        split_train=split_train,
        seed=random_state,
        device='cuda'
    )

    early_stopping = EarlyStopping(
        X_train, y_train, X_val, y_val,
        model_predict=lambda X: None, #cpd,
        get_model_weights=lambda: None, #lambda: cpd.node_states(),
        loss_fn=root_mean_squared_error_torch,
        abs_err=1e-6,
        rel_err=1e-4,
        early_stopping=early_stopping,
        verbose=verbose,
        start_degree=2
    )

    loss_fn = SquareBregFunction()

    time_start = time.time()

    for deg in range(2, max_degree + 1):
        tt_model = TensorTrainLayer(deg, rank, X_train.shape[1],
                               output_shape=1,
                               constrict_bond=False,
                               perturb=False,
                               seed=random_state).cuda()
        early_stopping.model_predict = tt_model
        early_stopping.get_model_weights = lambda: tt_model.node_states()
        converged = tt_model.tensor_network.accumulating_swipe(
            X_train, y_train, loss_fn,
            batch_size=-1,
            convergence_criterion=None,
            eps=np.geomspace(eps_start, eps_end, num=num_swipes*2).tolist(),
            method=method,
            skip_second=False,
            lr=1.0,
            orthonormalize=False,
            verbose=verbose == 1,
            num_swipes=num_swipes,
            direction='l2r',
            disable_tqdm=True
        )

        if early_stopping.convergence_criterion() or (time.time() - time_start) > timeout:
            break
    
    best_summary = early_stopping.best_summary()
    best_state_dict = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']

    singular = not converged

    # Restore best state if available
    tt_model = TensorTrainLayer(best_degree, rank, X_train.shape[1],
                                output_shape=1,
                                constrict_bond=False,
                                perturb=False,
                                seed=random_state).cuda()
    if best_state_dict is not None:
        tt_model.load_node_states(best_state_dict, set_value=True)
    y_pred_test = tt_full_predict(tt_model, X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    num_params = sum(p.tensor.numel() for i, p in enumerate(tt_model.tensor_network.train_nodes))
    val_history = early_stopping.val_history
    time_history = early_stopping.time_history
    return r2_test, rmse_test, singular, tt_model, best_degree, num_params, val_history, time_history



#%%
X_train, y_train, X_val, y_val, X_test, y_test = get_data(
    d=7,
    degree=10,
    num_train_points=50*70,
    num_val_points=50*70,
    num_test_points=10000,
    random_state=42,
    add_noise=0.0
)
#%%
tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    early_stopping=20,
    max_degree=20,
    rank=24,
    eps_start=1e-12,#1e4, #1e-10
    eps_end=1e-12,#1e-10, #1e-4
    #eps_start=1e-10,#1e4, #1e-10
    #eps_end=1e-4,#1e-10, #1e-4
    split_train=False,
    random_state=46,
    verbose=2
)
print(f"TT R2: {tt_r2}, RMSE: {tt_rmse}, Degree: {tt_degree}, Params: {tt_params}, Singular: {tt_singular}")

#%%
# Fit polynomial regression
poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank, poly_history, poly_time_history = evaluate_polynomial_regression(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree=10,
    d=5,
    abs_err=1e-5,
    rel_err=1e-4,
    early_stopping=10,
    verbose=2
)
print(f"Poly R2: {poly_r2}, RMSE: {poly_rmse}, Degree: {poly_degree}, Params: {poly_params}")
#%%
cpd_r2, cpd_rmse, cpd_singular, cpd_model, cpd_degree, cpd_params, cpd_history, cpd_time_history = evaluate_cpd(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    eps_start=1e-8,
    eps_end=1e-8,
    max_degree=10,
    rank=30,
    early_stopping=10,
    random_state=42,
    verbose=2
)
print(f"CPD R2: {cpd_r2}, RMSE: {cpd_rmse}, Degree: {cpd_degree}, Params: {cpd_params}, Singular: {cpd_singular}")
#%%
# Full tensor train fit
tt_full_r2, tt_full_rmse, tt_full_singular, tt_full_model, tt_full_degree, tt_full_params, tt_full_history, tt_full_time_history = evaluate_tt_full(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree=10,
    rank=6,
    early_stopping=10,
    random_state=42,
    verbose=2
)
print(f"TT Full R2: {tt_full_r2}, RMSE: {tt_full_rmse}, Degree: {tt_full_degree}, Params: {tt_full_params}, Singular: {tt_full_singular}")
#%%
# Evaluate growing CPD (by setting model_type='cpd' in evaluate_tensor_train)
cpd_grow_r2, cpd_grow_rmse, cpd_grow_singular, cpd_grow_model, cpd_grow_degree, cpd_grow_params, cpd_grow_history, cpd_grow_time_history = evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    early_stopping=5,
    max_degree=20,
    rank=100,
    eps_start=1e-12,
    eps_end=1e-12,
    split_train=False,
    method='ridge_cholesky',
    model_type='cpd',
    random_state=42,
    verbose=2
)
print(f"CPD Grow R2: {cpd_grow_r2}, RMSE: {cpd_grow_rmse}, Degree: {cpd_grow_degree}, Params: {cpd_grow_params}, Singular: {cpd_grow_singular}")
#%%
# Plot train and test
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Sort X_train and y_train for better visualization
sort_idx_train = np.argsort(X_train[:, 0])
X_train = X_train[sort_idx_train]
y_train = y_train[sort_idx_train]
# Sort X_test and y_test for better visualization
sort_idx_test = np.argsort(X_test[:, 0])
X_test = X_test[sort_idx_test]
y_test = y_test[sort_idx_test]
# Train set
ax[0].scatter(X_train, y_train, color='black', label='Train Data', alpha=0.5, marker='x')
y_tt_train_pred = tt_model.predict(X_train)
ax[0].scatter(X_train, y_tt_train_pred, color='orange', label='TT Prediction', alpha=0.5)
y_poly_train_pred = evaluate_poly(X_train, poly_coeffs, poly_model)
ax[0].scatter(X_train, y_poly_train_pred, color='blue', label='Poly Prediction', alpha=0.5)
y_cpd_train_pred = cpd_predict(cpd_model, X_train)
ax[0].scatter(X_train, y_cpd_train_pred, color='purple', label='CPD Prediction', alpha=0.5)
y_full_tt_train_pred = tt_full_predict(tt_full_model, X_train)
ax[0].scatter(X_train, y_full_tt_train_pred, color='green', label='Full TT Prediction', alpha=0.5)
y_cpd_grow_train_pred = cpd_grow_model.predict(X_train)
ax[0].scatter(X_train, y_cpd_grow_train_pred, color='red', label='CPD Grow Prediction', alpha=0.5)
ax[0].set_title('Train Set')
ax[0].set_xlabel('X')
ax[0].set_ylabel('y')
ax[0].legend()
# Test set
ax[1].plot(X_test, y_test, color='black', label='Test Data', alpha=0.5, linestyle='dashed')
y_test_pred = tt_model.predict(X_test)
ax[1].plot(X_test, y_test_pred, color='orange', label='TT Prediction', alpha=0.5)
y_poly_test_pred = evaluate_poly(X_test, poly_coeffs, poly_model)
ax[1].plot(X_test, y_poly_test_pred, color='blue', label='Poly Prediction', alpha=0.5)
y_cpd_test_pred = cpd_predict(cpd_model, X_test)
ax[1].plot(X_test, y_cpd_test_pred, color='purple', label='CPD Prediction', alpha=0.5)
y_full_tt_test_pred = tt_full_predict(tt_full_model, X_test)
ax[1].plot(X_test, y_full_tt_test_pred, color='green', label='Full TT Prediction', alpha=0.5)
y_cpd_grow_test_pred = cpd_grow_model.predict(X_test)
ax[1].plot(X_test, y_cpd_grow_test_pred, color='red', label='CPD Grow Prediction', alpha=0.5)
ax[1].set_title('Test Set')
ax[1].set_xlabel('X')
ax[1].set_ylabel('y')
ax[1].legend()

plt.tight_layout()
plt.show()
#%%
import pandas as pd
from tqdm import tqdm

eps = 3e-13
def collect_results(seeds=range(42, 42+20), degree=3, d=1, rank=24, cpd_rank=100, max_degree=10, eps=3e-13, one_value_only=False):
    rows = []
    n_values = [math.comb(i+d, d) for i in range(degree,degree+5)]
    if one_value_only:
        n_values = [n_values[0]]  # Only use the first value
    with tqdm(total=len(seeds) * len(n_values), desc="Collecting results") as pbar:
        for seed in seeds:
            for n in n_values:
                # Generate data once per seed and N
                X_train, y_train, X_val, y_val, X_test, y_test = get_data(
                    d=d,
                    degree=degree,
                    num_train_points=n,
                    num_val_points=n,
                    num_test_points=2, #test set is unused
                    random_state=seed
                )

                # Tensor Train model
                tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    early_stopping=max_degree,
                    max_degree=max_degree,
                    eps_start=eps,
                    eps_end=eps,
                    rank=rank, #change eps to 5e-12 to support higher rank
                    constrain_bond=False,
                    method='ridge_cholesky',
                    split_train=False,
                    random_state=seed,
                    verbose=0
                )
                for deg in tt_history.keys():
                    tt_loss = tt_history[deg]
                    tt_time = tt_time_history[deg]
                    rows.append((seed, "Growing TT", n, deg, float(tt_loss), tt_time, tt_r2, tt_rmse, tt_degree, tt_params))

                # # Polynomial baseline
                poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank, poly_history, poly_time_history = evaluate_polynomial_regression(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    max_degree=max_degree,
                    d=d,
                    abs_err=1e-5,
                    rel_err=1e-4,
                    early_stopping=max_degree,
                    verbose=0
                )
                for deg in poly_history.keys():
                    poly_loss = poly_history[deg]
                    poly_time = poly_time_history[deg]
                    rows.append((seed, "Poly.Reg.", n, deg, float(poly_loss), poly_time, poly_r2, poly_rmse, poly_degree, poly_params))

                # # CPD baseline
                cpd_r2, cpd_rmse, cpd_singular, cpd_model, cpd_degree, cpd_params, cpd_history, cpd_time_history = evaluate_cpd(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    max_degree=max_degree,
                    rank=cpd_rank,
                    eps_start=eps,
                    eps_end=eps,
                    early_stopping=max_degree,
                    split_train=False,
                    method='ridge_cholesky',
                    random_state=seed,
                    verbose=0
                )
                for deg in cpd_history.keys():
                    cpd_loss = cpd_history[deg]
                    cpd_time = cpd_time_history[deg]
                    rows.append((seed, "CPD", n, deg, float(cpd_loss), cpd_time, cpd_r2, cpd_rmse, cpd_degree, cpd_params))

                # Tensor Train without Growing
                tt_full_r2, tt_full_rmse, tt_full_singular, tt_full_model, tt_full_degree, tt_full_params, tt_full_history, tt_full_time_history = evaluate_tt_full(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    max_degree=max_degree,
                    rank=rank,
                    eps_start=eps,
                    eps_end=eps,
                    early_stopping=max_degree,
                    split_train=False,
                    method='ridge_cholesky',
                    random_state=seed,
                    verbose=0
                )
                for deg in tt_full_history.keys():
                    tt_full_loss = tt_full_history[deg]
                    tt_full_time = tt_full_time_history[deg]
                    rows.append((seed, "TT", n, deg, float(tt_full_loss), tt_full_time, tt_full_r2, tt_full_rmse, tt_full_degree, tt_full_params))

                # CPD with growing (using evaluate_tensor_train with model_type='cpd')
                cpd_grow_r2, cpd_grow_rmse, cpd_grow_singular, cpd_grow_model, cpd_grow_degree, cpd_grow_params, cpd_grow_history, cpd_grow_time_history = evaluate_tensor_train(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    early_stopping=max_degree,
                    max_degree=max_degree,
                    rank=cpd_rank,
                    eps_start=1e-12,
                    eps_end=1e-12,
                    split_train=False,
                    method='ridge_cholesky',
                    model_type='cpd',
                    random_state=seed,
                    verbose=0
                )
                for deg in cpd_grow_history.keys():
                    cpd_grow_loss = cpd_grow_history[deg]
                    cpd_grow_time = cpd_grow_time_history[deg]
                    rows.append((seed, "Growing CPD", n, deg, float(cpd_grow_loss), cpd_grow_time, cpd_grow_r2, cpd_grow_rmse, cpd_grow_degree, cpd_grow_params))

                pbar.update(1)

    df = pd.DataFrame(rows, columns=["seed", "name", "N", "degree", "loss", "time", "r2", "rmse", "best_degree", "num_params"])
    return df

# Example usage
# max_degree = 12
# d = 7
# degree = 3
# rank = 24
# cpd_rank = 200
# df = collect_results(d=d, degree=degree, rank=rank, cpd_rank=cpd_rank, eps=eps)
# print(df.head())
# print(len(df), "rows")
parameters = [
    {'degree': 3, 'max_degree': 8, 'd': 1, 'rank': 6, 'cpd_rank': 100},
    {'degree': 3, 'max_degree': 8, 'd': 3, 'rank': 12, 'cpd_rank': 100},
    {'degree': 3, 'max_degree': 8, 'd': 7, 'rank': 24, 'cpd_rank': 100},
    {'degree': 5, 'max_degree': 10, 'd': 1, 'rank': 6, 'cpd_rank': 100},
    {'degree': 5, 'max_degree': 10, 'd': 3, 'rank': 12, 'cpd_rank': 100},
    {'degree': 5, 'max_degree': 10, 'd': 7, 'rank': 24, 'cpd_rank': 100},
]
#%%
# MARK: Train here
# Collect the data and save the df to csv each time
all_dfs = []
for params in parameters:
    print(f"Collecting for params: {params}")
    df = collect_results(
        degree=params['degree'],
        d=params['d'],
        rank=params['rank'],
        cpd_rank=params['cpd_rank'],
        max_degree=params['max_degree'],
        eps=eps
    )
    df['max_degree'] = params['max_degree']
    all_dfs.append(df)
    # Save to csv
    df.to_csv(f"results_d{params['d']}_deg{params['degree']}_rank{params['rank']}_cpdrank{params['cpd_rank']}.csv", index=False)
#%%
# Now that all are saved, load them and plot them, one for each df:
# Load from csv
import pandas as pd
all_dfs = []
for params in parameters:
    df = pd.read_csv(f"results_d{params['d']}_deg{params['degree']}_rank{params['rank']}_cpdrank{params['cpd_rank']}.csv")
    all_dfs.append(df)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
for params, df in zip(parameters, all_dfs):
    plt.figure(figsize=(8, 6))
    sns.color_palette("tab10")
    sns.lineplot(
        data=df,
        x="degree",
        y="loss",
        style="N",
        hue="name",
        markers=True,
        errorbar="se",
        palette='tab10'
    )

    plt.yscale("log")
    plt.title(f"RMSE for dim={params['d']}, poly degree={params['degree']}, TT rank={params['rank']}, CPD rank={params['cpd_rank']}")
    plt.xlabel("Degree")
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    plt.show()

#%%
# Plot time vs degree as a bar plot
# Plot time only for the biggest number of points
df_N = df[df['N'] == df['N'].max()]
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_N,
    x="degree",
    y="time",
    hue="name"
)
plt.title(f"Training Time")
plt.xlabel("Degree")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.show()
#%%
#MARK: Rank plots
#%%
import pandas as pd
from tqdm import tqdm
max_degree = 10
d = 7
degree = 5
rank = 24
cpd_rank = 25
eps = 3e-13
ranks = [10,25,50,100,200,500]
def collect_results(seeds=range(42, 42+5), degree=3, d=1, eps=3e-13, one_value_only=False):
    rows = []
    n_values = [math.comb(i+d, d) for i in range(degree,degree+5)]
    if one_value_only:
        n_values = [n_values[0]]  # Only use the first value
    with tqdm(total=len(seeds) * len(n_values), desc="Collecting results") as pbar:
        for seed in seeds:
            for n in n_values:
                # Generate data once per seed and N
                X_train, y_train, X_val, y_val, X_test, y_test = get_data(
                    d=d,
                    degree=degree,
                    num_train_points=n,
                    num_val_points=n,
                    num_test_points=2, #test set is unused
                    random_state=seed
                )

                # Tensor Train model
                for r in ranks:
                    # tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
                    #     X_train, y_train,
                    #     X_val, y_val,
                    #     X_test, y_test,
                    #     early_stopping=max_degree,
                    #     max_degree=max_degree,
                    #     eps_start=eps,
                    #     eps_end=eps,
                    #     rank=r, #change eps to 5e-12 to support higher rank
                    #     constrain_bond=False,
                    #     method='ridge_cholesky',
                    #     split_train=False,
                    #     random_state=seed,
                    #     verbose=0
                    # )
                    # for deg in tt_history.keys():
                    #     tt_loss = tt_history[deg]
                    #     tt_time = tt_time_history[deg]
                    #     rows.append((seed, "tt", n, deg, float(tt_loss), tt_time, r))
                    # CPD with growing (using evaluate_tensor_train with model_type='cpd')
                    cpd_grow_r2, cpd_grow_rmse, cpd_grow_singular, cpd_grow_model, cpd_grow_degree, cpd_grow_params, cpd_grow_history, cpd_grow_time_history = evaluate_tensor_train(
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        early_stopping=max_degree,
                        max_degree=max_degree,
                        rank=r,
                        eps_start=1e-12,
                        eps_end=1e-12,
                        split_train=False,
                        method='ridge_cholesky',
                        model_type='cpd',
                        random_state=seed,
                        verbose=0
                    )
                    for deg in cpd_grow_history.keys():
                        cpd_grow_loss = cpd_grow_history[deg]
                        cpd_grow_time = cpd_grow_time_history[deg]
                        rows.append((seed, "Growing CPD", n, deg, float(cpd_grow_loss), cpd_grow_time, r, cpd_grow_r2, cpd_grow_rmse, cpd_grow_degree, cpd_grow_params))
                pbar.update(1)

    df = pd.DataFrame(rows, columns=["seed", "name", "N", "degree", "loss", "time", "rank", "r2", "rmse", "best_degree", "num_params"])
    return df

# Example usage
df = collect_results(d=d, degree=degree, eps=eps)
print(df.head())
print(len(df), "rows")
#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.color_palette("tab10")
sns.lineplot(
    data=df,
    x="degree",
    y="loss",
    style="N",
    hue="rank",
    markers=True,
    errorbar="se",
    palette='tab10'
)

plt.yscale("log")
#plt.title(f"RMSE for dim={d}, poly degree={degree}, TT rank={rank}, TT eps={eps}")
plt.xlabel("Degree")
plt.ylabel("Validation Loss")
plt.tight_layout()
plt.show()
# %%
import math

def num_tt_params(r, d, D):
    return (D-2)*r**2*d + 2*D*r*d

def num_poly_params(d, D):
    return math.comb(d + D, d)

def num_cpd_params(r, d, D):
    return r*d*D

# Make 5 plots for d = 1, 5, 10, 25, 50
# Plot r = 2, 6, 12, 24, 48 for TT and one poly in each
import matplotlib.pyplot as plt
import numpy as np

ds = [1, 3, 7, 10, 25]
ranks = [2, 6, 12, 24, 48]
cpd_ranks = [10, 25, 50, 100, 200]
for d in ds:
    plt.figure(figsize=(8, 6))
    D = np.arange(1, 21)
    for r in ranks:
        tt_params = [num_tt_params(r, d, deg) for deg in D]
        plt.plot(D, tt_params, label=f'TT rank={r}')
    for r in cpd_ranks:
        cpd_params = [num_cpd_params(r, d, deg) for deg in D]
        plt.plot(D, cpd_params, label=f'CPD rank={r}', linestyle='-.')
    poly_params = [num_poly_params(d, deg) for deg in D]
    plt.plot(D, poly_params, label='Poly.Reg.', linestyle='--', color='black')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Number of Parameters')
    plt.title(f'Number of Parameters vs Degree (d={d})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%
