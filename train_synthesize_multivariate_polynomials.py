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
        mask: float = 0.0,
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
    coeffs = torch.linalg.lstsq(Phi, y).solution  # (n_terms, 1)
    
    coeffs = coeffs.cpu().numpy()  # convert to numpy array

    return poly, coeffs

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

def get_data(d, degree, num_train_points, num_val_points, num_test_points, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(-2, 2, (num_train_points, d))
    X_val = rng.uniform(-2, 2, (num_val_points, d))
    X_test = rng.uniform(-2, 2, (num_test_points, d))

    poly = RandomPolynomial(
        d=d,
        degree=degree,
        sigma0=0.02,
        r=1,
        include_bias=True,
        interaction_only=False,
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
    split_train=False,
    random_state=42,
    verbose=0
):
    if max_degree > 2:
        tt = TensorTrainRegressorEarlyStopping(
            early_stopping=5,
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
    else:
        tt = None
        singular = True
        best_degree = np.nan
        r2_test = np.nan
        rmse_test = np.nan

    return r2_test, rmse_test, singular, tt, best_degree

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
    current = {'poly': None, 'coeffs': None}

    def model_predict(X):
        return evaluate_poly(X, current['coeffs'], current['poly'])

    def get_weights():
        return (current['poly'], current['coeffs'])

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

        poly, coeffs = fit_poly_mononomial(
            X_train, y_train, degree=deg, include_bias=True
        )

        current['poly'], current['coeffs'] = poly, coeffs

        if es.convergence_criterion():
            break

    best_summary = es.best_summary()
    best_poly, best_coeffs = best_summary['best_state_dict']
    best_degree = best_summary['best_degree']

    y_test_pred = evaluate_poly(X_test, best_coeffs, best_poly)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    return test_r2, test_rmse, best_poly, best_coeffs, best_degree

def evaluate_model(
    d,
    data_degree,
    num_train_points,
    num_val_points,
    num_test_points,
    max_degree,
    tt_rank,
    abs_err=1e-5,
    rel_err=1e-4,
    early_stopping=5,
    verbose=2,
    random_state=42
):
    # 1) generate data
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(
        d=d,
        degree=data_degree,
        num_train_points=num_train_points,
        num_val_points=num_val_points,
        num_test_points=num_test_points,
        random_state=random_state
    )

    # 2) fit & time Tensor Train
    torch.cuda.synchronize()
    t0 = time.time()
    tt_r2, tt_rmse, tt_singular, tt_model, tt_degree = evaluate_tensor_train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        max_degree=max_degree,
        rank=tt_rank,
        split_train=False,
        random_state=random_state,
        verbose=verbose
    )
    torch.cuda.synchronize()
    tt_time = time.time() - t0

    # 3) fit & time polynomial regression
    torch.cuda.synchronize()
    t1 = time.time()
    poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree = evaluate_polynomial_regression(
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

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_time,
        poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_time
    )

#%%
(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_time,
    poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_time
) = evaluate_model(
    d=10,
    data_degree=7,
    num_train_points=1000,
    num_val_points=300,
    num_test_points=1000,
    max_degree=8,
    tt_rank=25,
    abs_err=1e-5,
    rel_err=1e-4,
    early_stopping=5,
    verbose=2,
    random_state=42
)
print("Tensor Train Results:")
print(f"R2: {tt_r2:.4f}, RMSE: {tt_rmse:.4f}, Singular: {tt_singular}, Degree: {tt_degree}, Time: {tt_time:.2f}s")
print("Polynomial Regression Results:")
print(f"R2: {poly_r2:.4f}, RMSE: {poly_rmse:.4f}, Degree: {poly_degree}, Time: {poly_time:.2f}s")
#%%
from tqdm import tqdm
ds = [1, 2, 4, 8, 12]
data_degrees = [1, 2, 3, 4, 5, 6, 7, 10]
ranks = [3, 9, 27]
seeds = list(range(42, 52))  # 10 different seeds for robustness
results = [] #pairs for pandas df
# data_dim, data_degree, seed,
# tt_r2, tt_rmse, tt_degree, tt_time, tt_rank,
# poly_r2, poly_rmse, poly_degree, poly_time

tbar = tqdm(total=len(ds) * len(data_degrees) * len(seeds) * len(ranks), desc="Evaluating models")
for d in ds:
    for data_degree in data_degrees:
        for rank in ranks:
            for seed in seeds:
                tbar.set_description(f"Evaluating d={d}, degree={data_degree}, seed={seed}, rank={rank}")
                (
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    tt_r2, tt_rmse, tt_singular, tt_model, tt_degree, tt_time,
                    poly_r2, poly_rmse, poly_model, poly_coeffs, poly_degree, poly_time
                ) = evaluate_model(
                    d=d,
                    data_degree=data_degree,
                    num_train_points=10000,
                    num_val_points=3000,
                    num_test_points=10000,
                    max_degree=15,
                    tt_rank=rank,
                    abs_err=1e-5,
                    rel_err=1e-4,
                    early_stopping=5,
                    verbose=0,
                    random_state=seed
                )
                
                tbar.set_postfix({
                    'tt_r2': tt_r2,
                    'tt_rmse': tt_rmse,
                    'tt_degree': tt_degree,
                    'tt_time': tt_time,
                    'poly_r2': poly_r2,
                    'poly_rmse': poly_rmse,
                    'poly_degree': poly_degree,
                    'poly_time': poly_time
                })
                tbar.update(1)

                results.append((
                    d, data_degree, seed,
                    tt_r2, tt_rmse, tt_degree, tt_time, rank,
                    poly_r2, poly_rmse, poly_degree, poly_time
                ))

#%%
