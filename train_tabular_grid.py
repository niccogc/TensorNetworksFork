import os
import argparse
import json
import itertools
import copy
from train_tabular import train_model, load_tabular_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid search for tabular models')
    parser.add_argument('--grid_json', type=str, required=True, help='JSON file specifying the parameter grid')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--data_file', type=str, default='', help='Path to .pt file with {"X": X, "y": y}')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True)
    parser.add_argument('--model_type', type=str, choices=['tensor', 'xgboost', 'svm', 'mlp'], default='tensor')
    parser.add_argument('--layer_type', type=str, choices=['tt', 'operator'], default='tt', help='Layer type: tt (TensorTrainLayer) or operator (TensorOperatorLayer)')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm progress bars regardless of verbosity')
    parser.add_argument('--visualize_tn', type=str, default=None, help='Path to save tensor network visualization (e.g., .png)')

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

    # For simplicity, parse known args and leave the rest to be set by grid
    args, _ = parser.parse_known_args()

    # Load grid from YAML
    with open(args.grid_json, 'r') as f:
        grid = json.load(f)

    datasets = grid.pop('datasets', [args.data_file])
    param_names = list(grid.keys())
    param_values = [grid[k] for k in param_names]

    for dataset in datasets:
        run_args = copy.deepcopy(args)
        # If dataset is a filename, use as is; else, construct path
        if dataset.endswith('.pt'):
            run_args.data_file = dataset
        else:
            run_args.data_file = os.path.join(args.data_dir, dataset + '_tensor.pt')
        data = load_tabular_data(run_args.data_file, device=run_args.data_device)
        # For each grid combination
        for combo in itertools.product(*param_values):
            run_args_iter = copy.deepcopy(run_args)
            for k, v in dict(zip(param_names, combo)).items():
                setattr(run_args_iter, k, v)
            print(f"Running: {run_args_iter}")
            try:
                train_model(run_args_iter, data=data)
            except Exception as e:
                print(f"Error running {run_args_iter}: {e}")
                continue
