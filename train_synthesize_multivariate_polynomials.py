#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from tensor.bregman import SquareBregFunction
from tensor.module import TensorTrainRegressorEarlyStopping, EarlyStopping

from scipy import special
from sklearn.preprocessing import PolynomialFeatures
import math
import time

class RandomPolynomial:
    """
    Random multivariate polynomial of total degree ≤ D using sklearn PolynomialFeatures.

    Coefficients are sampled with per-degree scaling:
        sigma_k = sigma0 / ((k+1) * sqrt(C(d+k-1, k))) * r^{-k}

    - d: number of variables
    - degree: total degree D
    - sigma0: base scale for coefficients
    - r: typical input magnitude (if |x_j| ~ r, higher-degree terms are tamed)
    - include_bias: include the constant term
    - interaction_only: if True, monomials never have a power >1 on any variable
    - random_state: seed for reproducibility
    """
    def __init__(
        self,
        d: int,
        degree: int,
        sigma0: float = 0.2,
        r: float = 1.0,
        mask: float = 0.1,
        include_bias: bool = True,
        interaction_only: bool = False,
        random_state = None,
    ):
        self.d = int(d)
        self.degree = int(degree)
        self.sigma0 = float(sigma0)
        self.r = float(r)
        self.mask = float(mask)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        # RNG
        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        # Build PolynomialFeatures and "fit" on dummy data to populate .powers_
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        self.poly.fit(np.zeros((1, self.d)))  # just to get powers_

        # powers_: (n_features, d), each row is a multi-index alpha
        self._powers = self.poly.powers_  # ndarray of ints
        self._degrees = self._powers.sum(axis=1)  # (n_features,)

        # Precompute per-degree std
        self._deg_std = self._compute_degree_stds(self.d, self.degree, self.sigma0, self.r)

        # Sample coefficients
        self.coeffs_ = self._sample_coeffs()

    @staticmethod
    def _compute_degree_stds(d, D, sigma0, r):
        # sigma_k = sigma0 / ((k+1)*sqrt(C(d+k-1, k))) * r^{-k}
        deg_std = {}
        for k in range(D + 1):
            n_terms_k = math.comb(d + k - 1, k)  # number of monomials with total degree k
            sigma_k = sigma0 / ((k + 1) * math.sqrt(n_terms_k))
            if r != 0.0:
                sigma_k *= (r ** (-k))
            deg_std[k] = sigma_k
        return deg_std

    def _sample_coeffs(self):
        sigmas = np.array([self._deg_std[int(k)] for k in self._degrees], dtype=float)
        scale = self.rng.uniform(-10, 10, size=sigmas.shape)
        return np.exp(scale) * self.rng.normal(0, sigmas) * (1-self.rng.binomial(1, self.mask, size=sigmas.shape))

    def design_matrix(self, x: np.ndarray):
        """
        Return the monomial feature matrix Phi for x.

        x: (B, d) array
        Phi: (B, n_features) where n_features = C(D+d, d) (if include_bias=True; minus 1 otherwise)
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")
        return self.poly.transform(x)

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the polynomial at x.

        x: (B, d)
        returns y: (B,)
        """
        Phi = self.design_matrix(x)         # (B, n_features)
        return Phi.dot(self.coeffs_)        # (B,)
    
class RandomPolynomialRange:
    def __init__(self, d, degree, input_range=(-1, 1), random_state=None):
        self.d = d
        self.degree = degree
        self.range_start, self.range_end = input_range

        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        self.C = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.d,)
        )
        self.C = np.exp(self.C - np.max(self.C))
        self.C /= np.sum(self.C)
        self.roots = self.rng.uniform(
            low=self.range_start,
            high=self.range_end,
            size=(self.degree,)
        )

    def evaluate(self, x):
        """
        Evaluate the polynomial at x.

        x: (B, d) array
        returns y: (B,) array
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")
        
        t = np.dot(x, self.C)  # (B,) weighted sum of inputs

        # Output is a polynomial of t with roots at self.roots
        # Calculate distance to each root
        dist = (t[:, None] - self.roots[None, :])  # (B, degree)
        # Multiply by (t - root) for each root
        y = np.prod(dist, axis=1)  # (B,) product over roots
        
        return y



def get_data(d, degree, num_train_points, num_val_points, num_test_points, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(-1, 1, size=(num_train_points, d))
    X_val = rng.uniform(-1, 1, size=(num_val_points, d))
    X_test = rng.uniform(-1, 1, size=(num_test_points, d))

    # poly = RandomPolynomial(
    #     d=d,
    #     degree=degree,
    #     sigma0=0.02,
    #     r=1,
    #     include_bias=True,
    #     interaction_only=False,
    #     random_state=random_state
    # )
    poly = RandomPolynomialRange(
        d=d,
        degree=degree,
        input_range=(-1, 1),
        random_state=random_state
    )
    y_train = poly.evaluate(X_train)
    y_val = poly.evaluate(X_val)
    y_test = poly.evaluate(X_test)

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
    max_degree,      # formerly "carriages"
    rank,
    early_stopping=5,
    split_train=False,
    random_state=42,
    verbose=0
):
    if max_degree > 2:
        tt = TensorTrainRegressorEarlyStopping(
            early_stopping=early_stopping,
            rel_err=1e-4,
            abs_err=1e-6,
            validation_split=0.2,
            split_train=split_train,
            N=max_degree,
            r=rank,
            output_dim=1,
            linear_dim=X_train.shape[1] + 1,
            batch_size=-1,
            constrict_bond=True,
            seed=random_state,
            device='cuda',
            bf=SquareBregFunction(),
            lr=1.0,
            method='ridge_cholesky',
            verbose=verbose
        )
        try:
            # pass validation data into fit()
            tt.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            singular = tt._singular
        except Exception:
            singular = True

        # evaluate on the test set
        y_pred_test = tt.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        best_degree = tt._best_degree
        num_params = sum(p.tensor.numel() for i, p in enumerate(tt._model.tensor_network.train_nodes) if i < tt._best_degree)
    else:
        tt = None
        singular = True
        best_degree = np.nan
        r2_test = np.nan
        rmse_test = np.nan
        num_params = None

    return r2_test, rmse_test, singular, tt, best_degree, num_params

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
    if best_poly is None:
        return np.nan, np.nan, None, None, np.nan, None, None

    y_test_pred = evaluate_poly(X_test, best_coeffs, best_poly)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    num_params = np.prod(best_coeffs.shape)

    return test_r2, test_rmse, best_poly, best_coeffs, best_degree, num_params, best_rank

#%%
X_train, y_train, X_val, y_val, X_test, y_test = get_data(
    d=10,
    degree=10,
    num_train_points=10000,
    num_val_points=300,
    num_test_points=10000,
    random_state=46
)
tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params = evaluate_tensor_train(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    early_stopping=5,
    max_degree=15,
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
from tqdm import tqdm
features = [1, 2, 4, 8, 12]
poly_fit_degrees = [1, 2, 3, 4, 5, 6, 7, 10]
ranks = [4, 8, 24]
seeds = list(range(42, 52))  # 10 different seeds for robustness
results = [] #pairs for pandas df

num_train_points=1000
num_val_points=300
num_test_points=1000
max_degree=15
abs_err=1e-5
rel_err=1e-4
early_stopping=5
verbose=0
tbar = tqdm(total=len(features) * len(poly_fit_degrees) * len(seeds), desc="Evaluating models")
for d in features:
    for data_degree in poly_fit_degrees:
        for seed in seeds:
            # 1) generate data
            X_train, y_train, X_val, y_val, X_test, y_test = get_data(
                d=d,
                degree=data_degree,
                num_train_points=num_train_points,
                num_val_points=num_val_points,
                num_test_points=num_test_points,
                random_state=seed
            )

            for rank in ranks:
                tbar.set_description(f"Evaluating d={d}, degree={data_degree}, seed={seed}, rank={rank}")

                # 2) fit & time Tensor Train
                torch.cuda.synchronize()
                t0 = time.time()
                tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_params = evaluate_tensor_train(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    max_degree=max_degree,
                    rank=rank,
                    split_train=False,
                    random_state=seed,
                    verbose=verbose
                )
                torch.cuda.synchronize()
                tt_time = time.time() - t0
                results.append((
                    'tt', d, data_degree, seed, tt_r2, tt_rmse, tt_degree, tt_time, tt_params, rank,
                ))

            # 3) fit & time polynomial regression
            torch.cuda.synchronize()
            t1 = time.time()
            poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_params, poly_rank = evaluate_polynomial_regression(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                max_degree=max_degree,
                d=d,
                abs_err=abs_err,
                rel_err=rel_err,
                early_stopping=early_stopping,
                verbose=verbose
            )
            torch.cuda.synchronize()
            poly_time = time.time() - t1
            if poly_model is None:
                poly_time = np.nan
            results.append((
                'poly', d, data_degree, seed, poly_r2, poly_rmse, poly_degree, poly_time, poly_params, poly_rank
            ))
                
            tbar.set_postfix({
                'tt_r2': tt_r2,
                'tt_rmse': tt_rmse,
                'tt_degree': tt_degree,
                'tt_time': tt_time,
                'tt_params': tt_params,
                'poly_r2': poly_r2,
                'poly_rmse': poly_rmse,
                'poly_degree': poly_degree,
                'poly_time': poly_time,
                'poly_params': poly_params
            })
            tbar.update(1)


#%%
import pandas as pd
df = pd.DataFrame(results, columns=[
    'model', 'd', 'data_degree', 'seed',
    'r2', 'rmse', 'degree', 'time', 'params', 'rank'
])
# There are some "rank" for polynomials that are empty tensors, set these to -1 if they are isinstance(torch.Tensor)
df['rank'] = df['rank'].apply(lambda x: x if isinstance(x, int) else -1)
# Write CSV
df.to_csv('multivariate_polynomials_results.csv', index=False)
#%%
# Visualize some results
import pandas as pd
# Load the results
df = pd.read_csv('multivariate_polynomials_results.csv')

import seaborn as sns
import matplotlib.pyplot as plt
# Set the style
sns.set(style="whitegrid")
# First we plot the r2 scores as a function of number of parameters.
# We need to create a new variable (something like 'model_name') that combines model and rank for tt and just poly for poly
df['model_name'] = df.apply(lambda row: f"{row['model']}_rank{row['rank']}" if row['model'] == 'tt' else row['model'], axis=1)
# Also create a new variable that measures the difference between model degree and data degree
df['degree_diff'] = df['degree'] - df['data_degree']
#%%
# Get all lower R^2 than 0 and with d > 4
df_bad = df[(df['r2'] < 0)]
# Print all rows
print(df_bad.to_string(index=False))
#%%
# We can now group by model_name and seed to get the mean and sem of r2 scores
# Then we need to calculate the mean and sem across seeds.
df_plot = df.copy()
# Filter only for d = 1
df_plot = df_plot[df_plot['d'] == 4]
# Now we can plot the results
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_plot,
    x='params',
    y='r2',
    hue='model_name',
    marker='o',
    estimator='median',
    errorbar='se',
)
plt.ylim(0,1)
plt.xscale('log')
plt.legend(title='Model', loc='upper left')
plt.tight_layout()
plt.show()
# %%
# Create array that has d as the first dimension and data_degree as the second dimension, then one for each model_name.
# Then have each element be the mean time for that d and data_degree.
import numpy as np
# Get unique values of d and data_degree
d_values = df['d'].unique()
data_degree_values = df['data_degree'].unique()
model_names = df['model_name'].unique()
# Create a dictionary to hold the arrays
arrays = {model_name: np.zeros((len(d_values), len(data_degree_values))) for model_name in model_names}
# Fill the arrays with the mean time for each d and data_degree
for model_name in model_names:
    for i, d in enumerate(d_values):
        for j, data_degree in enumerate(data_degree_values):
            mean_time = df[(df['model_name'] == model_name) & (df['d'] == d) & (df['data_degree'] == data_degree)]['time'].mean()
            arrays[model_name][i, j] = mean_time
# Now we can plot the results using heatmap
fig, axs = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 6))
axs = axs.flatten() if len(model_names) > 1 else [axs]
for i, (model_name, array) in enumerate(arrays.items()):
    ax=axs[i]
    sns.heatmap(
        array,
        annot=True,
        fmt=".2f",
        xticklabels=data_degree_values,
        yticklabels=d_values,
        cmap='viridis',
        cbar_kws={'label': 'Mean Time (s)'},
        vmin=0,
        vmax=np.max(list(arrays.values())),
        ax=ax
    )
    ax.set_title(model_name)
    ax.set_xlabel('Data Degree')
    ax.set_ylabel('d (Number of Variables)')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/multivariate_polynomials_time_heatmap.png', dpi=300)
plt.show()
#%%
# Now do the same for r2 scores
arrays_r2 = {model_name: np.zeros((len(d_values), len(data_degree_values))) for model_name in model_names}
# Fill the arrays with the mean r2 for each d and data_degree
for model_name in model_names:
    for i, d in enumerate(d_values):
        for j, data_degree in enumerate(data_degree_values):
            mean_r2 = df[(df['model_name'] == model_name) & (df['d'] == d) & (df['data_degree'] == data_degree)]['r2'].mean()
            arrays_r2[model_name][i, j] = mean_r2
# Now we can plot the results using heatmap
fig, axs = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 6))
axs = axs.flatten() if len(model_names) > 1 else [axs]
for i, (model_name, array) in enumerate(arrays_r2.items()):
    ax=axs[i]
    sns.heatmap(
        array,
        annot=True,
        fmt=".2f",
        xticklabels=data_degree_values,
        yticklabels=d_values,
        cmap='viridis',
        cbar_kws={'label': 'Mean R2 Score'},
        vmin=0,
        vmax=1,
        ax=ax
    )
    ax.set_title(model_name)
    ax.set_xlabel('Data Degree')
    ax.set_ylabel('d (Number of Variables)')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/multivariate_polynomials_r2_heatmap.png', dpi=300)
plt.show()
# %%
