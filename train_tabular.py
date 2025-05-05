import os
import argparse
import torch
from models.xgboost import XGBRegWrapper, XGBClfWrapper
from models.svm import SVMRegWrapper, SVMClfWrapper
from models.mlp import MLPWrapper
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
    return X_train, y_train, X_val, y_val, X_test, y_test

def fit_model_fn(model_type, task, X_train, y_train):
    if model_type == 'tensor':
        return None  # Tensor model is handled in main for now
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def parse_xgb_args(args):
    xgb_params = {}
    for arg in vars(args):
        if arg.startswith('xgb.'):
            key = arg.split('xgb.')[1]
            xgb_params[key] = getattr(args, arg)
    return xgb_params

def evaluate_model(model, X, y, task, device=None):
    y_pred = model.predict(X)
    if task == 'classification':
        if y.ndim == 2:
            y_true = y.argmax(-1)
        else:
            y_true = y
        acc = accuracy_score(y_true, y_pred)
        return acc
    else:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return rmse

def train_model(args, data=None):
    args.version = '05-05-25'
    if data is None:
        data = load_tabular_data(args.data_file, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(args.data_file))[0]
    dataset_name = dataset_name.replace('_tensor', '')

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
    }
    tt_params = {
        'layer_type': args.tt_layer_type,
        'N': args.tt_N,
        'r': args.tt_r,
        'num_swipes': args.tt_num_swipes,
        'lr': args.tt_lr,
        'method': args.tt_method,
        'eps_min': args.tt_eps_min,
        'eps_max': args.tt_eps_max,
        'delta': args.tt_delta,
        'orthonormalize': args.tt_orthonormalize,
        'timeout': args.tt_timeout,
        'batch_size': args.tt_batch_size,
        'disable_tqdm': args.tt_disable_tqdm,
    }

    # WandB setup
    wandb_enabled = False
    config = vars(args)
    config['dataset_name'] = dataset_name
    if args.wandb_project:
        import wandb
        
        wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity)
        wandb_enabled = True

    # Save model type and hyperparameters to config
    if args.wandb_project:
        config_dict = vars(args)
        config_dict['model_type'] = args.model_type
        if args.model_type == 'xgboost':
            config_dict['xgboost_params'] = xgb_params
        elif args.model_type == 'svm':
            config_dict['svm_params'] = svm_params
        elif args.model_type == 'mlp':
            config_dict['mlp_params'] = mlp_params
        wandb.config.update(config_dict)

    # Model setup and training (unified for all models)
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    input_dim = X_train_np.shape[1]
    output_dim = y_train_np.shape[1]
    if args.model_type == 'xgboost':
        model = XGBRegWrapper(xgb_params) if args.task == 'regression' else XGBClfWrapper(xgb_params)
    elif args.model_type == 'svm':
        model = SVMRegWrapper(svm_params) if args.task == 'regression' else SVMClfWrapper(svm_params)
    elif args.model_type == 'mlp':
        model = MLPWrapper(input_dim, output_dim, mlp_params, task=args.task)
    elif args.model_type == 'tensor':
        torch.set_default_dtype(torch.float64)
        # Use torch tensors for tensor train
        model = TensorTrainWrapper(input_dim, output_dim, tt_params, task=args.task, device=args.device)
        X_train_tt = X_train.to(args.device)
        y_train_tt = y_train.to(args.device)
        converged = model.fit(X_train_tt, y_train_tt)
        if wandb_enabled:
            wandb.log({'singular': not converged})
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    if args.model_type != 'tensor':
        model.fit(X_train_np, y_train_np)
    # Unified evaluation
    if args.model_type == 'tensor':
        val_score = evaluate_model(model, X_val.to(args.device), y_val.to(args.device), args.task, device=args.device)
        test_score = evaluate_model(model, X_test.to(args.device), y_test.to(args.device), args.task, device=args.device)
    else:
        val_score = evaluate_model(model, X_val_np, y_val_np, args.task)
        test_score = evaluate_model(model, X_test_np, y_test_np, args.task)
    if args.task == 'classification':
        print('Validation Accuracy:', val_score)
        print('Test Accuracy:', test_score)
        if wandb_enabled:
            wandb.log({'val/accuracy': val_score, 'test/accuracy': test_score})
    else:
        print('Validation RMSE:', val_score)
        print('Test RMSE:', test_score)
        if wandb_enabled:
            wandb.log({'val/rmse': val_score, 'test/rmse': test_score})
    if wandb_enabled:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--data_file', type=str, required=True, help='Path to .pt file with {"X": X, "y": y}')
    parser.add_argument('--layer_type', type=str, choices=['tt', 'operator'], default='tt', help='Layer type: tt (TensorTrainLayer) or operator (TensorOperatorLayer)')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True, help='Task type: classification or regression')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm progress bars regardless of verbosity')
    parser.add_argument('--visualize_tn', type=str, default=None, help='Path to save tensor network visualization (e.g., .png)')
    parser.add_argument('--model_type', type=str, choices=['tensor', 'xgboost', 'svm', 'mlp'], default='tensor', help='Model type: tensor (TensorTrain/Operator), xgboost, svm, or mlp')

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
    parser.add_argument('--mlp_epochs', type=int, default=50, help='Number of epochs for MLP')
    parser.add_argument('--mlp_batch_size', type=int, default=128, help='Batch size for MLP')
    parser.add_argument('--mlp_device', type=str, default='cuda', help='Device for MLP')

    # Tensor Train hyperparameters
    parser.add_argument('--tt_layer_type', type=str, choices=['tt', 'operator'], default='tt', help='Layer type for tensor train')
    parser.add_argument('--tt_N', type=int, default=3, help='Number of carriages for tensor train')
    parser.add_argument('--tt_r', type=int, default=3, help='Bond dimension for tensor train')
    parser.add_argument('--tt_num_swipes', type=int, default=1, help='Number of swipes for tensor train')
    parser.add_argument('--tt_lr', type=float, default=1.0, help='Learning rate for tensor train')
    parser.add_argument('--tt_method', type=str, default='exact', help='Method for tensor train')
    parser.add_argument('--tt_eps_max', type=float, default=1.0, help='Initial Epsilon for tensor train')
    parser.add_argument('--tt_eps_min', type=float, default=1e-3, help='Final Epsilon for tensor train')
    parser.add_argument('--tt_delta', type=float, default=1.0, help='Delta for tensor train')
    parser.add_argument('--tt_orthonormalize', action='store_true', help='Orthonormalize for tensor train')
    parser.add_argument('--tt_timeout', type=float, default=None, help='Timeout for tensor train')
    parser.add_argument('--tt_batch_size', type=int, default=512, help='Batch size for tensor train')
    parser.add_argument('--tt_disable_tqdm', action='store_true', help='Disable tqdm for tensor train')

    args = parser.parse_args()
    train_model(args)  # loads data inside main by default
