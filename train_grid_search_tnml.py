from dotdict import DotDict
import torch
torch.set_default_dtype(torch.float64)

import numpy as np
from models.tnml import TNMLRegressor
from tensor.bregman import AutogradLoss
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
import pandas as pd

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
    model = TNMLRegressor(
        r=args.r,
        output_dim=output_dim,
        bf=bf,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_decay=args.eps_decay,
        batch_size=args.batch_size,
        method=args.method,
        num_swipes=args.num_swipes,
        model_type=args.model_type,
        task=args.task,
        verbose=args.verbose,
        early_stopping=args.early_stopping if args.early_stopping > 0 else None,
    )
    # Add num parameters to config
    model.fit(X_train, y_train, X_val, y_val)
    print(model._model.tensor_network.train_nodes)
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
    args = DotDict()
    paths = [
        ('/work3/aveno/Tabular/data/concrete_tensor.pt', 'regression'),
        ('/work3/aveno/Tabular/data/processed/gesture_tensor.pt', 'classification'),
        ('/work3/aveno/Tabular/data/energyprediction_tensor.pt', 'regression'),
        ('/work3/aveno/Tabular/data/processed/higgs_small_tensor.pt', 'classification'),
    ]
    args.device = 'cuda'
    args.data_device = 'cuda'
    args.model_type = 'tt'

    rs = [4, 8, 12]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.75

    args.batch_size = 1024
    args.early_stopping = 10

    args.verbose = 1
    args.method = 'ridge_exact'
    args.lin_dim = None

    seeds = list(range(42, 42+5))

    data = []
    for path, task in paths:
        dataset = path.split('/')[-1].replace('_tensor.pt', '')
        args.path = path
        args.task = task
        for r in rs:
            args.r = r
            for seed in seeds:
                args.seed = seed
                print(f"Training {dataset} with r={r}")
                result = train_model(args)
                data.append((dataset, np.nan, r, np.nan, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                print(f"Result: {result}")
    
    df = pd.DataFrame(data, columns=['dataset', 'N', 'r', 'lin_dim', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'converged_epoch', 'seed'])
    df['num_swipes'] = args.num_swipes
    df['eps_start'] = args.eps_start
    df['eps_decay'] = args.eps_decay
    df['early_stopping'] = args.early_stopping
    df['model_type'] = args.model_type
    df.to_csv('./results/grid_search_results_tnml.csv', index=False)
