import os
import argparse
import torch
from models.xgboost import XGBRegWrapper, XGBClfWrapper
from models.svm import SVMRegWrapper, SVMClfWrapper
from models.mlp import MLPWrapper
from models.polynomial_regression import PolynomialRegressionWrapper
from models.tensor_train import TensorTrainWrapper
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# ---- Tabular data loader ----
def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_val = data['X_val'].to(device)
    y_val = data['y_val'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    if 'processed' in filename:
        print("Processing data for tabular model...")
        X_train = torch.cat((X_train, torch.ones((X_train.shape[0], 1), device=X_train.device)), dim=-1)
        X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), device=X_val.device)), dim=-1)
        X_test = torch.cat((X_test, torch.ones((X_test.shape[0], 1), device=X_test.device)), dim=-1)
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y_true, metric='accuracy'):
    y_pred = model.predict(X)
    if metric == 'accuracy':
        if y_true.ndim == 2:
            y_true = y_true.argmax(-1)
        acc = accuracy_score(y_true.cpu().numpy(), y_pred)
        return acc
    elif metric == 'rmse':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        rmse = np.sqrt(mean_squared_error(y_true.cpu().numpy(), y_pred))
        return rmse
    elif metric == 'r2':
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        r2 = 1 - (np.sum((y_true.cpu().numpy() - y_pred) ** 2) / np.sum((y_true.cpu().numpy() - np.mean(y_true.cpu().numpy())) ** 2))
        return r2
    else:
        raise ValueError(f"Unknown metric: {metric}")

def train_model(args, data=None):
    args.version = '05-05-25v3'
    # WandB setup
    wandb_enabled = False
    if args.wandb_project:
        import wandb
        
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb_enabled = True

        # Update args with wandb config values
        for key, value in list(wandb.config.items()):
            setattr(args, key, value)
            print(f"WandB overriding parameter: {key} = {value}")
        
        # Then update the config with the args
        config_dict = vars(args)
        wandb.config.update(config_dict)
    
    if data is None:
        path = os.path.join(args.data_dir, args.dataset_name + '_tensor.pt')
        data = load_tabular_data(path, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # For each y, if it is not 2D, add a dimension
    if y_train.ndim == 1 and args.task == 'regression':
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    elif y_train.ndim == 1 and args.task == 'classification':
        num_classes = len(torch.unique(y_train.to(dtype=torch.long)))
        y_train = torch.nn.functional.one_hot(y_train.to(dtype=torch.long), num_classes=num_classes)
        y_val = torch.nn.functional.one_hot(y_val.to(dtype=torch.long), num_classes=num_classes)
        y_test = torch.nn.functional.one_hot(y_test.to(dtype=torch.long), num_classes=num_classes)

    xgb_params = {
        'n_estimators': args.xgb_n_estimators,
        'max_bin': args.xgb_max_bin,
        'learning_rate': args.xgb_learning_rate,
        'grow_policy': args.xgb_grow_policy,
        'tree_method': args.xgb_tree_method,
        'device': args.xgb_device,
        'n_jobs': args.xgb_n_jobs,
    }
    svm_params = {
        'C': args.svm_C,
        'kernel': args.svm_kernel,
        'gamma': args.svm_gamma,
    }
    mlp_params = {
        'hidden_layers': args.mlp_hidden_layers,
        'activation': args.mlp_activation,
        'lr': args.mlp_lr,
        'epochs': args.mlp_epochs,
        'batch_size': args.mlp_batch_size,
        'device': args.mlp_device,
        'type': args.mlp_type,
    }
    tt_params = {
        'layer_type': args.tt_layer_type,
        'N': args.tt_N,
        'r': args.tt_r,
        'num_swipes': args.tt_num_swipes,
        'lr': args.tt_lr,
        'method': args.tt_method,
        'lin_bond': args.tt_lin_bond,
        'lin_dim': args.tt_lin_dim,
        'verbose': args.tt_verbose,
        'eps_min': args.tt_eps_min,
        'eps_max': args.tt_eps_max,
        'orthonormalize': args.tt_orthonormalize,
        'timeout': args.tt_timeout,
        'batch_size': args.tt_batch_size,
        'disable_tqdm': args.tt_disable_tqdm or args.disable_tqdm,
        'early_stopping': args.tt_early_stopping,
        'track_eval': args.tt_track_eval,
        'save_every': args.tt_save_every,
    }

    # Add polynomial regression parameters
    poly_params = {
        'degree': args.poly_degree,
        'regularization': args.poly_regularization,
        'alpha': args.poly_alpha,
    }

    # If choosing SVM and the sample size is larger than 1000000, skip
    if args.model_type == 'svm' and X_train.shape[0] > 10000:
        print(f"Skipping SVM training for {args.dataset_name} due to large sample size.")
        if wandb_enabled:
            wandb.finish()
            wandb.teardown()
        return

    # Save model type and hyperparameters to config
    if args.wandb_project:
        config_dict = {}
        config_dict['model_type'] = args.model_type
        if args.model_type == 'xgboost':
            config_dict['xgboost_params'] = xgb_params
        elif args.model_type == 'svm':
            config_dict['svm_params'] = svm_params
        elif args.model_type == 'mlp':
            config_dict['mlp_params'] = mlp_params
        wandb.config.update(config_dict)

    # Model setup and training (unified for all models)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    if args.model_type == 'xgboost':
        model = XGBRegWrapper(xgb_params) if args.task == 'regression' else XGBClfWrapper(xgb_params)
    elif args.model_type == 'svm':
        model = SVMRegWrapper(svm_params) if args.task == 'regression' else SVMClfWrapper(svm_params)
    elif args.model_type == 'mlp':
        model = MLPWrapper(input_dim, output_dim, mlp_params, task=args.task)
        # Add num parameters to config
        if wandb_enabled:
            wandb.config.update({'num_parameters': sum([p.numel() for p in model.model.parameters()])})
    elif args.model_type == 'tensor':
        torch.set_default_dtype(torch.float64)
        X_train = X_train.to(torch.float64)
        y_train = y_train.to(torch.float64)
        X_val = X_val.to(torch.float64)
        y_val = y_val.to(torch.float64)
        X_test = X_test.to(torch.float64)
        y_test = y_test.to(torch.float64)
        # Use torch tensors for tensor train
        model = TensorTrainWrapper(input_dim, output_dim, tt_params, task=args.task, device=args.device)
        # Add num parameters to config
        if wandb_enabled:
            wandb.config.update({'num_parameters': model.model.num_parameters()})
        converged = model.fit(X_train, y_train, X_val if args.tt_track_eval else None, y_val if args.tt_track_eval else None)
        if wandb_enabled:
            wandb.log({'singular': not converged})
    elif args.model_type == 'polynomial':
        model = PolynomialRegressionWrapper(
            degree=poly_params['degree'],
            regularization=poly_params['regularization'],
            alpha=poly_params['alpha']
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    if args.model_type != 'tensor' and args.model_type != 'polynomial':
        model.fit(X_train, y_train)
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)
    test_score = evaluate_model(model, X_test, y_test, metric)
    if args.task == 'classification':
        print('Validation Accuracy:', val_score)
        print('Test Accuracy:', test_score)
        if wandb_enabled:
            wandb.log({'val/accuracy': val_score, 'test/accuracy': test_score})
    else:
        print('Validation RMSE:', val_score)
        print('Test RMSE:', test_score)
        # Calculate R2 score as well
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        r2_test = evaluate_model(model, X_test, y_test, metric='r2')
        print('Validation R2:', r2_val)
        print('Test R2:', r2_test)
        if wandb_enabled:
            wandb.log({'val/rmse': val_score, 'test/rmse': test_score, 'val/r2': r2_val, 'test/r2': r2_test})
    if wandb_enabled:
        wandb.finish()
        wandb.teardown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (without .pt extension)')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True, help='Task type: classification or regression')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm progress bars regardless of verbosity')
    parser.add_argument('--model_type', type=str, choices=['tensor', 'xgboost', 'svm', 'mlp', 'polynomial'], default='tensor', help='Model type: tensor (TensorTrain/Operator), xgboost, svm, mlp, or polynomial')

    # XGBoost hyperparameters
    parser.add_argument('--xgb_n_estimators', type=int, default=300, help='Number of boosting rounds for XGBoost')
    parser.add_argument('--xgb_max_bin', type=int, default=100, help='Max bin for XGBoost')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.3, help='Learning rate for XGBoost')
    parser.add_argument('--xgb_grow_policy', type=str, default='depthwise', help='Grow policy for XGBoost')
    parser.add_argument('--xgb_tree_method', type=str, default='hist', help='Tree method for XGBoost')
    parser.add_argument('--xgb_device', type=str, default='cuda', help='Device for XGBoost')
    parser.add_argument('--xgb_n_jobs', type=int, default=-1, help='Number of parallel jobs for XGBoost')

    # SVM hyperparameters
    parser.add_argument('--svm_C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='Kernel type for SVM')
    parser.add_argument('--svm_gamma', type=str, default='scale', help='Kernel coefficient for SVM')

    # MLP hyperparameters
    parser.add_argument('--mlp_hidden_layers', type=int, nargs='+', default=[64, 64], help='Hidden layer sizes for MLP')
    parser.add_argument('--mlp_activation', type=str, default='relu', help='Activation function for MLP')
    parser.add_argument('--mlp_lr', type=float, default=1e-3, help='Learning rate for MLP')
    parser.add_argument('--mlp_epochs', type=int, default=25, help='Number of epochs for MLP')
    parser.add_argument('--mlp_batch_size', type=int, default=512, help='Batch size for MLP')
    parser.add_argument('--mlp_device', type=str, default='cuda', help='Device for MLP')
    parser.add_argument('--mlp_type', type=str, default='standard', choices=['standard', 'residual', 'pinet'], help='MLP type: standard, residual, or pinet')

    # Tensor Train hyperparameters
    parser.add_argument('--tt_layer_type', type=str, choices=['tt', 'operator', 'linear'], default='tt', help='Layer type for tensor train')
    parser.add_argument('--tt_N', type=int, default=3, help='Number of carriages for tensor train')
    parser.add_argument('--tt_r', type=int, default=3, help='Bond dimension for tensor train')
    parser.add_argument('--tt_num_swipes', type=int, default=1, help='Number of swipes for tensor train')
    parser.add_argument('--tt_lr', type=float, default=1.0, help='Learning rate for tensor train')
    parser.add_argument('--tt_method', type=str, default='exact', help='Method for tensor train')
    parser.add_argument('--tt_eps_max', type=float, default=1.0, help='Initial Epsilon for tensor train')
    parser.add_argument('--tt_eps_min', type=float, default=1e-3, help='Final Epsilon for tensor train')
    parser.add_argument('--tt_CB', type=int, default=4, help='Convolution bond for tensor train')
    parser.add_argument('--tt_orthonormalize', action='store_true', help='Orthonormalize for tensor train')
    parser.add_argument('--tt_timeout', type=float, default=None, help='Timeout for tensor train')
    parser.add_argument('--tt_batch_size', type=int, default=512, help='Batch size for tensor train')
    parser.add_argument('--tt_verbose', type=int, default=2, help='Verbosity level for tensor train')
    parser.add_argument('--tt_disable_tqdm', action='store_true', help='Disable tqdm for tensor train')
    parser.add_argument('--tt_lin_bond', type=int, default=1, help='Bond dimension for linear transform in tensor train')
    parser.add_argument('--tt_lin_dim', type=float, default=1.0, help='Output dimension for linear transform in tensor train')
    parser.add_argument('--tt_early_stopping', type=int, default=0, help='Early stopping patience for tensor train')
    parser.add_argument('--tt_track_eval', action='store_true', help='Track evaluation during training for tensor train')
    parser.add_argument('--tt_save_every', type=int, default=10, help='Save model every N epochs for tensor train')

    # Polynomial Regression hyperparameters
    parser.add_argument('--poly_degree', type=int, default=2, help='Degree of polynomial features')
    parser.add_argument('--poly_regularization', type=str, choices=['l1', 'l2', None], default=None, help='Regularization type for polynomial regression')
    parser.add_argument('--poly_alpha', type=float, default=1.0, help='Regularization strength for polynomial regression')

    args = parser.parse_args()
    train_model(args)  # loads data inside main by default
