import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from functools import partial
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from tensor.bregman import SquareBregFunction
from tensor.module import TensorTrainRegressor

import numpy as np
import sklearn.preprocessing as skpp
import sklearn.datasets as skds
import argparse

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

def main(project_name=None, entity=None):
    wandb.init(project=project_name, entity=entity)
    conf = wandb.config
    X, y = load_data()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross validate with polynomial features
    r2_scores_tt = []
    rmse_scores_tt = []
    wandb_log = {}
    singular = False
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_val_cv, y_val_cv = X[val_idx], y[val_idx]

        X_quant = QuantileTransformer(output_distribution="uniform", subsample=1_000_000, random_state=0)
        X_train_cv = X_quant.fit_transform(X_train_cv)
        X_val_cv = X_quant.transform(X_val_cv)

        tensor_train = TensorTrainRegressor(
            N=conf.N,
            r=conf.r,
            output_dim=1,
            linear_dim=conf.linear_dim,
            batch_size=conf.batch_size,
            perturb=conf.perturb,
            constrict_bond=conf.constrict_bond,
            seed=42,
            device='cuda',
            bf=SquareBregFunction(),
            lr=1.0,
            eps_start=conf.eps_start,
            eps_end= conf.eps_end,
            method='ridge_cholesky',
            num_swipes=conf.num_swipes,
            verbose=0
        )
        try:
            tensor_train.fit(X_train_cv, y_train_cv)
        except Exception as e:
            singular = True
            break

        y_pred_cv = tensor_train.predict(X_val_cv)
        
        tt_score = r2_score(y_val_cv, y_pred_cv)
        r2_scores_tt.append(tt_score)
        wandb_log[f"r2_score_fold_{fold + 1}"] = tt_score
        
        tt_rmse = root_mean_squared_error(y_val_cv, y_pred_cv)
        rmse_scores_tt.append(tt_rmse)
        wandb_log[f"rmse_fold_{fold + 1}"] = tt_rmse
        
        print(f"Validation R2: {tt_score:.4f}, RMSE: {tt_rmse:.4f}")
    if singular:
        print("Model failed to fit due to singular matrix.")
        wandb_log["r2"] = None
        wandb_log["rmse"] = None
        wandb_log["r2_low"] = None
        wandb_log["r2_high"] = None
        wandb_log["rmse_low"] = None
        wandb_log["rmse_high"] = None
        wandb.log(wandb_log)
        return
    print(f"5-fold CV R2: {np.mean(r2_scores_tt):.4f} ± {np.std(r2_scores_tt) / np.sqrt(len(r2_scores_tt)):.4f}")
    print(f"5-fold CV RMSE: {np.mean(rmse_scores_tt):.4f} ± {np.std(rmse_scores_tt) / np.sqrt(len(rmse_scores_tt)):.4f}")
    wandb_log["r2"] = np.mean(r2_scores_tt)
    wandb_log["rmse"] = np.mean(rmse_scores_tt)
    wandb_log["r2_low"] = np.mean(r2_scores_tt) - np.std(r2_scores_tt) / np.sqrt(len(r2_scores_tt))
    wandb_log["r2_high"] = np.mean(r2_scores_tt) + np.std(r2_scores_tt) / np.sqrt(len(r2_scores_tt))
    wandb_log["rmse_low"] = np.mean(rmse_scores_tt) - np.std(rmse_scores_tt) / np.sqrt(len(rmse_scores_tt))
    wandb_log["rmse_high"] = np.mean(rmse_scores_tt) + np.std(rmse_scores_tt) / np.sqrt(len(rmse_scores_tt))
    wandb.log(wandb_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep for Tensor Train Regressor")
    parser.add_argument("--project_name", type=str, default="house_sweeps", help="W&B project name")
    parser.add_argument("--entity", type=str, default="tensorGang", help="W&B entity name")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep ID to resume")
    parser.add_argument("--start_sweep_only", action='store_true', help="Only start the sweep without running the agent")
    args = parser.parse_args()


    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "r2_low"},
        "parameters": {
            "N": {
                "min": 2,
                "max": 16,
                "distribution": "int_uniform"
            },
            "r": {
                "min": 2,
                "max": 24,
                "distribution": "int_uniform"
            },
            "linear_dim": {
                "min": 2,
                "max": 17,
                "distribution": "int_uniform"
            },
            "eps_start": {
                "distribution": "log_uniform_values",
                "min": 1e-12,
                "max": 1e2
            },
            "eps_end": {
                "distribution": "log_uniform_values",
                "min": 1e-12,
                "max": 1
            },
            "batch_size": {
                "value": 512
            },
            "num_swipes": {
                "min": 1,
                "max": 10,
                "distribution": "int_uniform"
            },
            "perturb": {
                "value": True
            },
            "constrict_bond": {
                "value": True
            }
        },
    }

    if not args.sweep_id:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name, entity=args.entity)
        print("Starting sweep with ID:", sweep_id)
    else:
        sweep_id = args.sweep_id
        print("Adding agent to existing sweep with ID:", sweep_id)
    if args.start_sweep_only:
        exit(0)

    main = partial(main, project_name=args.project_name, entity=args.entity)

    wandb.agent(sweep_id, function=main, project=args.project_name, entity=args.entity)