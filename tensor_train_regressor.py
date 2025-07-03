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

x_train, y_train, x_val, y_val, x_test, y_test = load_tabular_data('/work3/aveno/Tabular/data/processed/house_tensor.pt', device='cuda')

x_std, x_mean = torch.std_mean(x_train, dim=0, unbiased=False, keepdim=True)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
x_val = (x_val - x_mean) / x_std

x_min, x_max = x_train.amin(dim=0, keepdim=True), x_train.amax(dim=0, keepdim=True)

eps_val = 1e-2

x_test_mask = ((x_min <= x_test) & (x_test <= x_max)).all(-1)
x_val_mask = ((x_min <= x_val) & (x_val <= x_max)).all(-1)

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

# Combine x_train and x_val again
x_train = torch.concat((x_train, x_val), dim=0)
y_train = torch.concat((y_train, y_val), dim=0)
#%%
from tensor.bregman import SquareBregFunction
from tensor.layers import TensorTrainLayer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
import torch
import numpy as np

class TensorTrainRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, N=2, r=2, input_dim=None, output_dim=1, 
                 constrict_bond=True, perturb=True, seed=42,
                 device='cuda', bf=None,
                 lr=1.0, eps_start=1e-12, eps_end=1e-12, eps_r=0.5, 
                 batch_size=512, method='ridge_cholesky',
                 num_swipes=5,
                 verbose=0):
        self.N = N
        self.r = r
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.constrict_bond = constrict_bond
        self.perturb = perturb
        self.seed = seed
        self.device = device
        self.bf = bf if bf is not None else SquareBregFunction()
        self.lr = lr
        self.epss = np.geomspace(eps_start, eps_end, 2 * num_swipes).tolist() if eps_end != eps_start else [eps_end] * (2 * num_swipes)
        self.eps_r = eps_r
        self.batch_size = batch_size
        self.method = method
        self.num_swipes = num_swipes
        self.verbose = verbose
        
        self._model = None
        
    def _initialize_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set")
        self._model = TensorTrainLayer(self.N, self.r, self.input_dim, 
                                       output_shape=self.output_dim,
                                       constrict_bond=self.constrict_bond,
                                       perturb=self.perturb,
                                       seed=self.seed).to(self.device)
    
    def fit(self, X, y):
        # X, y: numpy arrays or torch tensors
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        
        if self._model is None:
            self.input_dim = X.shape[1]
            self._initialize_model()
        
        # define convergence criterion function for progress printing
        def convergence_criterion():
            y_pred_train = self._model(X)
            rmse = torch.sqrt(torch.mean((y_pred_train - y)**2))
            if self.verbose > 0:
                print('Train RMSE:', rmse.item())
            return False
        
        # Call accumulating_swipe
        self._model.tensor_network.accumulating_swipe(
            X, y, self.bf,
            batch_size=self.batch_size,
            lr=self.lr,
            eps=self.epss,
            eps_r=self.eps_r,
            convergence_criterion=convergence_criterion,
            orthonormalize=False,
            method=self.method,
            verbose=self.verbose,
            num_swipes=self.num_swipes,
            skip_second=False,
            direction='l2r',
            disable_tqdm=True
        )
        
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        y_pred = self._model(X)
        return y_pred.detach().cpu().numpy()
    
    def score(self, X, y):
        # Return R2 score on X, y
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        y_pred = self._model(X)
        ss_res = torch.sum((y_pred - y)**2)
        ss_tot = torch.sum((y - torch.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()
    
#%%
import numpy as np
from sklearn.model_selection import KFold
from time import time

# Suppose X, y are numpy arrays
X = x_train.cpu().numpy()
y = y_train.cpu().numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
start = time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train_cv, y_train_cv = X[train_idx], y[train_idx]
    X_val_cv, y_val_cv = X[val_idx], y[val_idx]

    reg = TensorTrainRegressor(
        N=3,
        r=4,
        input_dim=X.shape[1],
        output_dim=1,
        seed=42,
        device='cuda',
        verbose=0,
        num_swipes=4,
        perturb=True,
        eps_start = 5,
        eps_end = 1e-3
    )
    reg.fit(X_train_cv, y_train_cv)
    
    score = r2_score(y_val_cv, reg.predict(X_val_cv))
    print(f"Validation R2: {score:.4f}")
    r2_scores.append(score)
end = time()
print(f"Time taken for 5-fold CV: {end - start:.2f}")
print(f"Average 5-fold CV R2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores) / np.sqrt(len(r2_scores)):.4f}")
# Print score on train set as well
train_score = r2_score(y, reg.predict(X))
print(f"Train R2: {train_score:.4f}")
#%%
# Try with default degree 3 polynomial fit with L2 regularization
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Cross validate with polynomial features
r2_scores_poly = []

start = time()
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train_cv, y_train_cv = X[train_idx], y[train_idx]
    X_val_cv, y_val_cv = X[val_idx], y[val_idx]

    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_cv)
    X_val_poly = poly.transform(X_val_cv)

    model = Ridge(alpha=1000)
    model.fit(X_train_poly, y_train_cv)

    score = r2_score(y_val_cv, model.predict(X_val_poly))
    print(f"Validation R2: {score:.4f}")
    r2_scores_poly.append(score)
end = time()
print(f"Time taken for 5-fold CV (Polynomial with L2): {end - start:.2f}")
print(f"Average 5-fold CV R2 (Polynomial with L2): {np.mean(r2_scores_poly):.4f} ± {np.std(r2_scores_poly) / np.sqrt(len(r2_scores_poly)):.4f}")
# Print score on train set as well
train_score_poly = r2_score(y, model.predict(poly.transform(X)))
print(f"Train R2 (Polynomial with L2): {train_score_poly:.4f}")
#%%