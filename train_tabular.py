import argparse
import os
import torch
from torch.nn import functional as F
from tensor.layers import TensorTrainLayer, TensorOperatorLayer
from tensor.bregman import XEAutogradBregman, SquareBregFunction
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from collections import Counter

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

def main():
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--data_file', type=str, required=True, help='Path to .pt file with {"X": X, "y": y}')
    parser.add_argument('--layer_type', type=str, choices=['tt', 'operator'], default='tt', help='Layer type: tt (TensorTrainLayer) or operator (TensorOperatorLayer)')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True, help='Task type: classification or regression')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=3, help='Number of carriages')
    parser.add_argument('--r', type=int, default=3, help='Bond dimension')
    parser.add_argument('--num_swipes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='exact')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--orthonormalize', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
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

    args = parser.parse_args()

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

    # WandB setup
    wandb_enabled = False
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb_enabled = True

    # Set CUDA device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_device = torch.device(args.data_device if torch.cuda.is_available() or args.data_device == 'cpu' else 'cpu')

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_data(args.data_file, data_device)
    N, f = X_train.shape
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(-1)
    l = y_train.shape[1]
    if args.task == 'classification':
        output_shape = (l-1,) if l > 1 else (1,)
    else:
        output_shape = (l,)

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

    # Model setup
    if args.model_type == 'tensor':
        torch.set_default_dtype(torch.float64)
        if args.layer_type == 'tt':
            layer = TensorTrainLayer(
                num_carriages=args.N,
                bond_dim=args.r,
                input_features=f,
                output_shape=output_shape
            ).to(device)
        else:
            # Construct operator as in cum_sum_operator.py
            ops = []
            p = f
            N = args.N
            for n in range(N):
                H = torch.triu(torch.ones((1 if n == 0 else p, p), dtype=torch.get_default_dtype(), device=device), diagonal=0)
                D = torch.zeros((p, p, p, 1 if n == N-1 else p), dtype=torch.get_default_dtype(), device=device)
                for i in range(p):
                    D[i, i, i, 0 if n == N-1 else i] = 1
                C = torch.einsum('ij,j...->i...', H, D)
                ops.append(C)
            layer = TensorOperatorLayer(
                operator=ops,
                input_features=f,
                bond_dim=args.r,
                num_carriages=args.N,
                output_shape=output_shape
            ).to(device)
        print('Num params:', layer.num_parameters())

        # Visualize tensor network if requested
        if args.visualize_tn:
            from tensor.utils import visualize_tensornetwork
            import matplotlib.pyplot as plt
            visualize_tensornetwork(layer.tensor_network)
            plt.savefig(args.visualize_tn)
            plt.close()
            print(f"Tensor network visualization saved to {args.visualize_tn}")

        # Bregman function
        X_train_for_bregman = X_train[:64].to(device)
        with torch.inference_mode():
            y_pred = layer(X_train_for_bregman)
            w = 1/y_pred.std().item() if y_pred.std().item() > 0 else 1.0
            del y_pred
        if args.task == 'classification':
            bf = XEAutogradBregman(w=w)
        else:
            bf = SquareBregFunction()

        # Move validation and test sets to model device for evaluation
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        X_test = X_test.to(device)
        y_test = X_test.to(device)

        def convergence_criterion(_, __):
            # Always use validation set for convergence check
            y_pred_val = layer(X_val)
            if args.task == 'classification':
                y_pred_val = torch.cat((y_pred_val, torch.zeros_like(y_pred_val[:, :1])), dim=1) if y_pred_val.shape[1] == y_val.shape[1] - 1 else y_pred_val
                y_val_idx = y_val.argmax(dim=-1).cpu().numpy()
                y_pred_idx = y_pred_val.argmax(dim=-1).cpu().numpy()
                balanced_acc = balanced_accuracy_score(y_val_idx, y_pred_idx)
                print("Validation Balanced Accuracy:", balanced_acc)
                if wandb_enabled:
                    wandb.log({"val/b_acc": balanced_acc})
            else:
                mse = F.mse_loss(y_pred_val, y_val).item()
                print("Validation MSE:", mse)
                if wandb_enabled:
                    wandb.log({"val/mse": mse})
            return False

        # Training
        try:
            layer.tensor_network.accumulating_swipe(
                X_train, y_train, bf,
                batch_size=args.batch_size,
                num_swipes=args.num_swipes,
                method=args.method,
                lr=args.lr,
                eps=args.eps,
                delta=args.delta,
                orthonormalize=args.orthonormalize,
                convergence_criterion=convergence_criterion,
                timeout=args.timeout,
                verbose=2,
                data_device=data_device,
                model_device=device,
                disable_tqdm=args.disable_tqdm
            )
        except Exception as e:
            print('Training failed:', e)

        # Final test set evaluation (always runs)
        y_pred_test = layer(X_test)
        if args.task == 'classification':
            y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1) if y_pred_test.shape[1] == y_test.shape[1] - 1 else y_pred_test
            accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
            print('Test Acc:', accuracy_test)
            if wandb_enabled:
                wandb.log({"test/b_acc_f": accuracy_test})
        else:
            mse_test = F.mse_loss(y_pred_test, y_test).item()
            print('Test MSE:', mse_test)
            if wandb_enabled:
                wandb.log({"test/mse_f": mse_test})
    else:
        # XGBoost, SVM, or MLP models
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy().squeeze()
        X_val_np = X_val.cpu().numpy()
        y_val_np = y_val.cpu().numpy().squeeze()
        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy().squeeze()
        if args.model_type == 'xgboost':
            from models.xgboost import XGBRegWrapper, XGBClfWrapper
            if args.task == 'regression':
                model = XGBRegWrapper(xgb_params)
            else:
                model = XGBClfWrapper(xgb_params)
            model.fit(X_train_np, y_train_np)
        elif args.model_type == 'svm':
            from models.svm import SVMRegWrapper, SVMClfWrapper
            if args.task == 'regression':
                model = SVMRegWrapper(svm_params)
            else:
                model = SVMClfWrapper(svm_params)
            model.fit(X_train_np, y_train_np)
        elif args.model_type == 'mlp':
            from models.mlp import MLPWrapper
            input_dim = X_train_np.shape[1]
            output_dim = 1 if y_train_np.ndim == 1 else y_train_np.shape[1]
            model = MLPWrapper(input_dim, output_dim, mlp_params, task=args.task)
            # Update config with MLP parameters
            if args.wandb_project:
                wandb.config.update({'mlp_num_params': sum(p.numel() for p in model.model.parameters())})
            print({'mlp_num_params': sum(p.numel() for p in model.model.parameters())})
            model.fit(X_train_np, y_train_np)
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")
        val_pred = model.predict(X_val_np)
        test_pred = model.predict(X_test_np)
        if args.task == 'classification':
            from sklearn.metrics import balanced_accuracy_score
            val_score = balanced_accuracy_score(y_val_np.argmax(-1), val_pred)
            test_score = balanced_accuracy_score(y_test_np.argmax(-1), test_pred)
            print('Validation Balanced Accuracy:', val_score)
            print('Test Balanced Accuracy:', test_score)
            if wandb_enabled:
                wandb.log({'val/b_acc': val_score, 'test/b_acc_f': test_score})
        else:
            from sklearn.metrics import mean_squared_error
            val_mse = mean_squared_error(y_val_np, val_pred)
            test_mse = mean_squared_error(y_test_np, test_pred)
            print('Validation MSE:', val_mse)
            print('Test MSE:', test_mse)
            if wandb_enabled:
                wandb.log({'val/mse': val_mse, 'test/mse_f': test_mse})

if __name__ == '__main__':
    main()
