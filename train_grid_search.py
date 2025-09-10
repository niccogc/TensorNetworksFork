import os
import argparse
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from models.tensor_train import TensorTrainRegressor
from tensor.bregman import AutogradLoss
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score

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

def train_model(args, data=None):
    if data is None:
        data = load_tabular_data(args.path, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # For each y, if it is not 2D, add a dimension
    if y_train.ndim == 1 and args.task == 'regression':
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    elif (y_train.ndim == 1 or y_train.shape[1] == 1) and args.task == 'classification':
        num_classes = len(torch.unique(y_train.to(dtype=torch.long)))
        y_train = torch.nn.functional.one_hot(y_train.to(dtype=torch.long), num_classes=num_classes).squeeze(1)
        y_val = torch.nn.functional.one_hot(y_val.to(dtype=torch.long), num_classes=num_classes).squeeze(1)
        y_test = torch.nn.functional.one_hot(y_test.to(dtype=torch.long), num_classes=num_classes).squeeze(1)

    # Model setup and training (unified for all models)
    output_dim = y_train.shape[1]
        
    X_train = X_train.to(torch.float64)
    y_train = y_train.argmax(dim=-1) if args.task == 'classification' else y_train.to(torch.float64)
    X_val = X_val.to(torch.float64)
    y_val = y_val.argmax(dim=-1) if args.task == 'classification' else y_val.to(torch.float64)
    X_test = X_test.to(torch.float64)
    y_test = y_test.argmax(dim=-1) if args.task == 'classification' else y_test.to(torch.float64)
    
    if args.task == 'regression':
        bf = AutogradLoss(torch.nn.MSELoss(reduction='none')) 
    else:
        bf = AutogradLoss(torch.nn.CrossEntropyLoss(reduction='none'))

    # Use torch tensors for tensor train
    model = TensorTrainRegressor(
        N=args.N,
        r=args.r,
        output_dim=output_dim,
        linear_dim=args.lin_dim,
        bf=bf,
        constrict_bond=False,
        perturb=False,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_decay=args.eps_decay,
        batch_size=args.batch_size,
        method=args.method,
        num_swipes=args.num_swipes,
        model_type=args.model_type,
        cum_sum=args.cum_sum,
        task=args.task,
        verbose=args.verbose,
        early_stopping=args.early_stopping if args.early_stopping > 0 else None,
    )
    # Add num parameters to config
    model.fit(X_train, y_train, X_val, y_val)
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = model._model.num_parameters()
    converged_epoch = model._early_stopper.epoch

    if args.task == 'classification':
        return {'val_rmse': np.nan, 'val_r2': np.nan, 'val_accuracy': val_score, 'num_params': num_params, 'converged_epoch': converged_epoch}
    else:
        # Calculate R2 score as well
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        return {'val_rmse': val_score, 'val_r2': r2_val, 'val_accuracy': np.nan, 'num_params': num_params, 'converged_epoch': converged_epoch}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (without .pt extension)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--model_type', type=str, default='tt', required=True, help='Type of model to train: tt, cpd, _type1, etc.')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Task type: regression or classification')

    # Tensor Train hyperparameters
    parser.add_argument('--N', type=int, default=3, help='Number of carriages for tensor train')
    parser.add_argument('--r', type=int, default=3, help='Bond dimension for tensor train')
    parser.add_argument('--num_swipes', type=int, default=30, help='Number of swipes for tensor train')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate for tensor train')
    parser.add_argument('--method', type=str, default='ridge_exact', help='Method for tensor train')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Initial Epsilon for tensor train')
    parser.add_argument('--eps_decay', type=float, default=0.75, help='Epsilon decay factor for tensor train')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for tensor train')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity level for tensor train')
    parser.add_argument('--lin_dim', type=int, default=None, help='Linear dimension for tensor train (if any)')
    parser.add_argument('--cum_sum', action='store_true', help='Use cumulative sum layer instead of tensor train')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience for tensor train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    args.path = os.path.join(args.data_dir, args.dataset_name + '_tensor.pt')
    result = train_model(args)  # loads data inside main by default
    print(result)
