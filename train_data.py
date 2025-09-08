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
from tensor.module import TensorTrainRegressor, EarlyStopping, TensorTrainRegressorEarlyStopping

from sklearn.preprocessing import QuantileTransformer

def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    X_train = data['X_train'].cpu().numpy()
    y_train = data['y_train'].cpu().numpy()
    X_val = data['X_val'].cpu().numpy()
    y_val = data['y_val'].cpu().numpy()
    X_test = data['X_test'].cpu().numpy()
    y_test = data['y_test'].cpu().numpy()
    if 'processed' not in filename:
        print("Processing data for tabular model...")
        X_train = X_train[..., :-1]
        X_val = X_val[..., :-1]
        X_test = X_test[..., :-1]
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_data('/work3/aveno/Tabular/data/processed/house_tensor.pt', device='cuda')

quantile_transformer = QuantileTransformer(output_distribution="uniform", random_state=42)
X_train = quantile_transformer.fit_transform(X_train)
X_val = quantile_transformer.transform(X_val)
X_test = quantile_transformer.transform(X_test)
#%%
tt = TensorTrainRegressor(
    num_swipes=15,
    eps_start=1.0,
    eps_end=100.0,
    N=14,
    r=14,
    linear_dim=8,
    output_dim=1,
    batch_size=2048,
    constrict_bond=False,
    perturb=False,
    seed=42,
    device='cuda',
    bf=AutogradLoss(torch.nn.MSELoss(reduction='none')),
    lr=1.0,
    method="ridge_trace",
    model_type="tt", #tt_type1_bias_first_no_train_linear
    verbose=1 # 1
)
tt.fit(X_train, y_train, X_val, y_val)
# evaluate on the test set
y_pred_test = tt.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
if y_pred_test.shape[1] > 1:
    accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred_test, axis=1))
else:
    accuracy = np.nan
print("R2 test:", r2_test, "RMSE test:", rmse_test, "Accuracy:", accuracy)
# %%
import matplotlib.pyplot as plt
import numpy as np
epochs = []
rmses = []
for traj in tt.trajectory:
    epochs.append(traj['epoch'])
    rmses.append(traj['val_rmse'])
rmses = np.array(rmses)
plt.plot(epochs, rmses)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Validation RMSE')
plt.title('Validation RMSE over Epochs')
# %%
tt = TensorTrainRegressorEarlyStopping(
    num_swipes=1,
    eps_start=1/180,
    eps_end=1/180,
    early_stopping=15,
    rel_err=1e-4,
    abs_err=1e-6,
    N=100,
    r=8,
    output_dim=1,
    batch_size=2048,
    seed=42,
    device='cuda',
    bf=AutogradLoss(torch.nn.MSELoss(reduction='none')),
    lr=1.0,
    method="ridge_trace",
    model_type="tt", #tt_type1_bias_first_no_train_linear
    verbose=2
)
#try:
# pass validation data into fit()
tt.fit(X_train, y_train, X_val=X_val, y_val=y_val)
# evaluate on the test set
y_pred_test = tt.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
if y_pred_test.shape[1] > 1:
    accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred_test, axis=1))
else:
    accuracy = np.nan
print("R2 test:", r2_test, "RMSE test:", rmse_test, "Accuracy:", accuracy)
# %%
