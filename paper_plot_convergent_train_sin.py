#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from tensor.bregman import SquareBregFunction
from tensor.module import TensorTrainRegressorEarlyStopping, EarlyStopping

import math
from scipy import special
from sklearn.preprocessing import PolynomialFeatures

def compute_y_from_x(X: np.ndarray, frequency) -> np.ndarray:
    return np.cos(X * frequency)[:, 0]

def get_data(num_train_points, num_val_points, num_test_points, frequency=3, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(-math.pi/2, math.pi/2, size=(num_train_points, 1)) #rng.normal(1, 0.1, size=(num_train_points, d))
    X_val = rng.uniform(-math.pi/2, math.pi/2, size=(num_val_points, 1)) #rng.normal(1, 0.1, size=(num_val_points, d))
    X_test = rng.uniform(-math.pi/2, math.pi/2, size=(num_test_points, 1)) #rng.normal(1, 0.3, size=(num_test_points, d))

    y_train = compute_y_from_x(X_train, frequency=frequency)
    y_val = compute_y_from_x(X_val, frequency=frequency)
    y_test = compute_y_from_x(X_test, frequency=frequency)

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    eps_start=1e-12,
    eps_end=1e-12,
    max_degree=10,      # formerly "carriages"
    rank=25,
    early_stopping=5,
    split_train=False,
    random_state=42,
    verbose=0
):
    if max_degree > 2:
        tt = TensorTrainRegressorEarlyStopping(
            eps_start=eps_start,
            eps_end=eps_end,
            early_stopping=early_stopping,
            rel_err=1e-4,
            abs_err=1e-6,
            split_train=split_train,
            N=max_degree,
            r=rank,
            output_dim=1,
            batch_size=-1,
            constrict_bond=False,
            seed=random_state,
            device='cuda',
            bf=SquareBregFunction(),
            lr=1.0,
            method='ridge_cholesky',
            verbose=verbose
        )
        #try:
        # pass validation data into fit()
        tt.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        singular = tt._singular
        # except Exception as e:
        #     singular = True
        #     print(f"Error during fitting: {e}")

        # evaluate on the test set
        y_pred_test = tt.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        best_degree = tt._best_degree
        num_params = sum(p.tensor.numel() for i, p in enumerate(tt._model.tensor_network.train_nodes) if i < tt._best_degree)
        val_history = tt._early_stopping.val_history
        time_history = tt._early_stopping.time_history
    else:
        tt = None
        singular = True
        best_degree = np.nan
        r2_test = np.nan
        rmse_test = np.nan
        num_params = None
        val_history = None

    return r2_test, rmse_test, singular, tt, best_degree, num_params, val_history, time_history

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
        if special.comb(d + deg, d) >= 500_000:
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

#%%
X_train, y_train, X_val, y_val, X_test, y_test = get_data(
    num_train_points=10000,
    num_val_points=1000,
    num_test_points=1000,
    random_state=42
)
tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    early_stopping=10,
    max_degree=30,
    rank=25,
    split_train=False,
    random_state=46,
    verbose=2
)
print(f"TT R2: {tt_r2}, RMSE: {tt_rmse}, Degree: {tt_degree}, Params: {tt_params}, Singular: {tt_singular}")
#%%
# Plot train and test
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Train set
ax[0].scatter(X_train, y_train, color='blue', label='Train Data', alpha=0.5)
y_train_pred = tt_model.predict(X_train)
ax[0].scatter(X_train, y_train_pred, color='red', label='TT Prediction', alpha=0.5)
ax[0].set_title('Train Set')
ax[0].set_xlabel('X')
ax[0].set_ylabel('y')
ax[0].legend()
# Test set
ax[1].scatter(X_test, y_test, color='green', label='Test Data', alpha=0.5)
y_test_pred = tt_model.predict(X_test)
ax[1].scatter(X_test, y_test_pred, color='orange', label='TT Prediction', alpha=0.5)
ax[1].set_title('Test Set')
ax[1].set_xlabel('X')
ax[1].set_ylabel('y')
ax[1].legend()

plt.tight_layout()
plt.show()
#%%
# Fit polynomial regression
poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank, poly_history, poly_time_history = evaluate_polynomial_regression(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    max_degree=20,
    d=5,
    abs_err=1e-5,
    rel_err=1e-4,
    early_stopping=5,
    verbose=2
)
print(f"Poly R2: {poly_r2}, RMSE: {poly_rmse}, Degree: {poly_degree}, Params: {poly_params}")
#%%
import pandas as pd
from tqdm import tqdm
from time import time
# MARK: Change d here, 1, 3, 5
def collect_results(seeds=range(42, 42+5)):
    rows = []
    frequencys = [1, 3, 5, 7]
    with tqdm(total=len(seeds) * len(frequencys), desc="Collecting results") as pbar:
        for seed in seeds:
            for frequency in frequencys:
                # Generate data once per seed and N
                X_train, y_train, X_val, y_val, X_test, y_test = get_data(
                    num_train_points=500,
                    num_val_points=1000,
                    num_test_points=2, # test set is unused
                    frequency=frequency,
                    random_state=seed
                )

                # Tensor Train model
                tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params, tt_history, tt_time_history = evaluate_tensor_train(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    eps_start=2e-12,
                    eps_end=2e-12,
                    early_stopping=50,
                    max_degree=50,
                    rank=35,
                    split_train=False,
                    random_state=seed,
                    verbose=0
                )

                # Polynomial baseline
                poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank, poly_history, poly_time_history = evaluate_polynomial_regression(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    max_degree=50,
                    d=1,
                    abs_err=1e-5,
                    rel_err=1e-4,
                    early_stopping=50,
                    verbose=0
                )

                # tt_history and poly_history are assumed to be dicts: {degree: validation_loss}
                for deg in tt_history.keys():
                    tt_loss = tt_history[deg]
                    tt_time = tt_time_history[deg]
                    rows.append((seed, "tt", tt_time, frequency, deg, float(tt_loss)))

                for deg in poly_history.keys():
                    poly_loss = poly_history[deg]
                    poly_time = poly_time_history[deg]
                    rows.append((seed, "poly", poly_time, frequency, deg, float(poly_loss)))
                pbar.update(1)

    df = pd.DataFrame(rows, columns=["seed", "name", 'time', "frequency", "degree", "loss"])
    return df

# Example usage
df = collect_results()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df,
    x="degree",
    y="loss",
    hue="name",
    style="frequency",
    markers=True,
    errorbar="se"
)

plt.xscale("log")
plt.yscale("log")
plt.title("Validation Loss with Mean ± SEM")
plt.xlabel("Degree")
plt.ylabel("Validation Loss")
plt.tight_layout()
plt.show()

#%%
# Plot time vs degree as a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df,
    x="degree",
    y="time",
    hue="name"
)
# %%
