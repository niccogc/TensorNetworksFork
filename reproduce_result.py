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

from tensor.module import TensorTrainRegressor
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
        linear_dim=17,
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