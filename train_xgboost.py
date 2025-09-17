import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
import xgboost as xgb

# ---- Tabular data loader ----
def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_val = data['X_val'].to(device)
    y_val = data['y_val'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    if torch.all(X_train[:, -1] == 1) and torch.all(X_val[:, -1] == 1) and torch.all(X_test[:, -1] == 1):
        X_train = X_train[:, :-1]
        X_val = X_val[:, :-1]
        X_test = X_test[:, :-1]

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y_true, metric='accuracy'):
    y_pred = model.predict(X)
    if metric == 'accuracy':
        if y_true.ndim == 2:
            y_true = y_true.argmax(-1)
        y_true = y_true.cpu().numpy()
        acc = accuracy_score(y_true, y_pred.argmax(-1))
        return acc
    elif metric == 'rmse':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.cpu().numpy()
        rmse = root_mean_squared_error(y_true, y_pred)
        return rmse
    elif metric == 'r2':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        y_true = y_true.cpu().numpy()
        r2 = r2_score(y_true, y_pred)
        return r2
    else:
        raise ValueError(f"Unknown metric: {metric}")

# --- XGBoost Wrapper ---
class XGBoost:
    """
    A unified wrapper for XGBoost that handles both regression and classification tasks.
    """
    def __init__(self, task='regression', xgb_params=None):
        self.task = task
        if xgb_params is None:
            xgb_params = {}

        # Use GPU for XGBoost if available and device is 'cuda'
        if torch.cuda.is_available() and xgb_params.get('device') == 'cuda':
             xgb_params['tree_method'] = 'hist'
        
        if self.task == 'regression':
            self.model = xgb.XGBRegressor(**xgb_params)
        elif self.task == 'classification':
            self.model = xgb.XGBClassifier(eval_metric='mlogloss', **xgb_params)
        else:
            raise ValueError(f"Unknown task: {task}")

    def fit(self, X, y):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        if self.task == 'classification' and y_np.ndim == 2:
            y_np = y_np.argmax(axis=-1)

        self.model.fit(X_np, y_np)

    def predict(self, X):
        X_np = X.cpu().numpy()
        if self.task == 'classification':
            return self.model.predict_proba(X_np)
        else:
            return self.model.predict(X_np)
    
    @property
    def _model(self):
        # A property to access the underlying model for compatibility, e.g., for num_parameters.
        # This part is a bit of a placeholder as XGBoost doesn't have a direct .num_parameters() method.
        # We can return a dictionary of important attributes instead.
        return self
    
    def num_parameters(self):
        return np.nan

def train_model(args, data=None):
    if data is None:
        data = load_tabular_data(args.path, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Model setup
    if args.model_type == 'xgboost':
        xgb_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.lr,
            'seed': args.seed,
            'device': args.device  # Pass device to XGBoost
        }
        model = XGBoost(task=args.task, xgb_params=xgb_params)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = model._model.num_parameters()
    converged_epoch = np.nan # Not applicable for XGBoost in this setup

    if args.task == 'classification':
        return {'val_rmse': np.nan, 'val_r2': np.nan, 'val_accuracy': val_score, 'num_params': num_params, 'converged_epoch': converged_epoch}
    else:
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        return {'val_rmse': val_score, 'val_r2': r2_val, 'val_accuracy': np.nan, 'num_params': num_params, 'converged_epoch': converged_epoch}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost Training for Tabular Data')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (without .pt extension)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training (for XGBoost, enables GPU usage)')
    parser.add_argument('--data_device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device where the dataset is stored')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Task type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost'], help='Model to train')

    # XGBoost hyperparameters
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for XGBoost')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum tree depth')
    
    # Note: batch_size is not used by this XGBoost implementation
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (not used for XGBoost)')

    args = parser.parse_args()
    args.path = os.path.join(args.data_dir, args.dataset_name + '_tensor.pt')
    
    # Set seed for reproducibility
    np.random.seed(args.seed)

    result = train_model(args)
    print(result)