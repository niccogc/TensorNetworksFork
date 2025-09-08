#%%
import numpy as np
import sklearn.preprocessing as skpp
import sklearn.datasets as skds

def load_openml(name, y_dict = None):
    df = skds.fetch_openml(name=name,as_frame=True)
    X = df.data.to_numpy(dtype=np.float64)     
    y = df.target
    if y_dict is not None: y=y.astype("str").map(y_dict)
    y = y.to_numpy(dtype=np.float64)
    return X, y

def load_data():
    X, y = load_openml("house_16H")
    
    if len(y.shape)==1: 
        y = y[:,np.newaxis]
    
    y_scaler = skpp.StandardScaler()
    y = y_scaler.fit_transform(y)

    return X, y

X, y = load_data()
#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross validate with polynomial features
r2_scores_poly = []
rmse_scores_poly = []
alpha = 1 # [1e-8, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train_cv, y_train_cv = X[train_idx], y[train_idx]
    X_val_cv, y_val_cv = X[val_idx], y[val_idx]

    X_quant = QuantileTransformer(output_distribution="uniform",subsample=1_000_000,random_state=0)
    X_train_cv = X_quant.fit_transform(X_train_cv)
    X_val_cv = X_quant.transform(X_val_cv)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_cv)
    X_val_poly = poly.transform(X_val_cv)

    model = Ridge(alpha=alpha) if alpha > 0.0 else LinearRegression()
    model.fit(X_train_poly, y_train_cv)

    y_pred_cv = model.predict(X_val_poly)
    score = r2_score(y_val_cv, y_pred_cv)
    print(f"Validation R2: {score:.4f}")
    rmse = np.sqrt(np.mean((y_val_cv - y_pred_cv) ** 2))
    print(f"Validation RMSE: {rmse:.4f}")
    r2_scores_poly.append(score)
    rmse_scores_poly.append(rmse)

print(f"Average 5-fold CV R2 (Polynomial with L2): {np.mean(r2_scores_poly):.4f} ± {np.std(r2_scores_poly) / np.sqrt(len(r2_scores_poly)):.4f}")
print(f"Average 5-fold CV RMSE (Polynomial with L2): {np.mean(rmse_scores_poly):.4f} ± {np.std(rmse_scores_poly) / np.sqrt(len(rmse_scores_poly)):.4f}")
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

        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        
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
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model(X)
        return y_pred.detach().cpu().numpy()
    
    def score(self, X, y):
        # Return R2 score on X, y
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model(X)
        ss_res = torch.sum((y_pred - y)**2)
        ss_tot = torch.sum((y - torch.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()
#%%
import torch
torch.set_default_dtype(torch.float64)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross validate
r2_scores_poly = []
rmse_scores_poly = [] 
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train_cv, y_train_cv = X[train_idx], y[train_idx]
    X_val_cv, y_val_cv = X[val_idx], y[val_idx]

    X_quant = QuantileTransformer(output_distribution="uniform",subsample=1_000_000,random_state=0)
    X_train_cv = X_quant.fit_transform(X_train_cv)
    X_val_cv = X_quant.transform(X_val_cv)

    tensor_train = TensorTrainRegressor(
        N=8,
        r=12,
        input_dim=X_train_cv.shape[1],
        output_dim=1,
        seed=42,
        device='cuda',
        bf=SquareBregFunction(),
        lr=1.0,
        eps_start=1,
        eps_end=1e-2,
        method='ridge_cholesky',
        num_swipes=10,
        verbose=0
    )
    tensor_train.fit(X_train_cv, y_train_cv)

    y_pred_cv = tensor_train.predict(X_val_cv)

    score = r2_score(y_val_cv, y_pred_cv)
    print(f"Validation R2: {score:.4f}")
    rmse = root_mean_squared_error(y_val_cv, y_pred_cv)
    print(f"Validation RMSE: {rmse:.4f}")
    r2_scores_poly.append(score)
    rmse_scores_poly.append(rmse)

print(f"Average 5-fold CV R2 (Polynomial with L2): {np.mean(r2_scores_poly):.4f} ± {np.std(r2_scores_poly) / np.sqrt(len(r2_scores_poly)):.4f}")
print(f"Average 5-fold CV RMSE (Polynomial with L2): {np.mean(rmse_scores_poly):.4f} ± {np.std(rmse_scores_poly) / np.sqrt(len(rmse_scores_poly)):.4f}")
#%%