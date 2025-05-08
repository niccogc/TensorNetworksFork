import wandb
import os
import argparse
from functools import partial
from train_tabular import train_model, load_tabular_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid search for tabular models')
    parser.add_argument('--sweep_id', type=str, required=True, help='WandB sweep ID')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset (without .pt extension)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True)
    parser.add_argument('--model_type', type=str, choices=['tensor', 'xgboost', 'svm', 'mlp'], default='tensor')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable tqdm progress bars regardless of verbosity')

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
    parser.add_argument('--tt_delta', type=float, default=1.0, help='Delta for tensor train')
    parser.add_argument('--tt_CB', type=int, default=4, help='Convolution bond for tensor train')
    parser.add_argument('--tt_orthonormalize', action='store_true', help='Orthonormalize for tensor train')
    parser.add_argument('--tt_timeout', type=float, default=0, help='Timeout for tensor train')
    parser.add_argument('--tt_batch_size', type=int, default=512, help='Batch size for tensor train')
    parser.add_argument('--tt_verbose', type=int, default=2, help='Verbosity level for tensor train')
    parser.add_argument('--tt_disable_tqdm', action='store_true', help='Disable tqdm for tensor train')
    parser.add_argument('--tt_lin_bond', type=int, default=1, help='Bond dimension for linear transform in tensor train')
    parser.add_argument('--tt_lin_dim', type=float, default=1.0, help='Output dimension for linear transform in tensor train')
    parser.add_argument('--tt_early_stopping', type=int, default=0, help='Early stopping patience for tensor train')
    parser.add_argument('--tt_track_eval', action='store_true', help='Track evaluation during training for tensor train')
    parser.add_argument('--tt_save_every', type=int, default=0, help='Save model every N epochs for tensor train')

    args, _ = parser.parse_known_args()

    # Hardcode some arguments
    args.disable_tqdm = True
    args.tt_disable_tqdm = True
    if args.tt_save_every == 0:
        args.tt_save_every = 5
    if args.tt_timeout == 0:
        args.timeout = 600


    path = os.path.join(args.data_dir, args.dataset_name + '_tensor.pt')
    data = load_tabular_data(path, device=args.data_device)

    func = partial(train_model, args, data=data)
    wandb.agent(sweep_id=args.sweep_id, function=func, entity=args.wandb_entity, project=args.wandb_project)