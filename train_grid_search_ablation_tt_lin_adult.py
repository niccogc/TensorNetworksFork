from datetime import datetime
from dotdict import DotDict
import torch
torch.set_default_dtype(torch.float64)
from train_grid_search import train_model
from load_ucirepo import get_ucidata
import pandas as pd
import numpy as np

datasets = [
  ('adult', 2, 'classification'),
]

if __name__ == '__main__':
    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'
    args.model_type = 'tt_lin'

    Ns = [2, 3, 4, 5, 6]
    rs = [4, 8, 12, 16]
    lin_dims = [0.25, 0.5, 0.75]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.25

    args.batch_size = 1024
    args.early_stopping = 10

    args.verbose = 1
    args.method = 'ridge_cholesky'

    seeds = list(range(42, 42+5))

    for dataset, dataset_id, task in datasets:
        results = []
        data = get_ucidata(dataset_id, task, args.data_device)
        num_features = data[0].shape[1]
        args.task = task
        for N in Ns:
            for r in rs:
                for lin_dim in lin_dims:
                    if num_features > 50 and r > 10:
                        continue
                    for seed in seeds:
                        try:
                            args.N = N
                            args.r = r
                            args.lin_dim = lin_dim
                            args.seed = seed
                            print(f"Training {dataset} with N={args.N}, r={r}, lin_dim={lin_dim}")
                            result = train_model(args, data=data, test=False)
                            results.append((dataset, args.N, args.r, args.lin_dim, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], args.seed))
                            print(f"Result: {result}")
                        except:
                            print("Failed, skipping...")
                            torch.cuda.empty_cache()
                            continue
    
        df = pd.DataFrame(results, columns=['dataset', 'N', 'r', 'lin_dim', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'converged_epoch', 'seed'])
        df['num_swipes'] = args.num_swipes
        df['eps_start'] = args.eps_start
        df['eps_decay'] = args.eps_decay
        df['early_stopping'] = args.early_stopping
        df['model_type'] = args.model_type

        if len(df) == 0:
            exit(0)

        df.to_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv', index=False)

        # Take the best one and run it on the test set
        # First we aggregate over seeds to find the best (N, r) pair
        df_agg = df.groupby(['N', 'r', 'lin_dim']).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()
        if task == 'regression':
            best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
        else:
            best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]
        
        args.N = int(best_row['N'])
        args.r = int(best_row['r'])
        args.lin_dim = float(best_row['lin_dim'])
        
        args.seed = 1337  # Fixed seed for final evaluation
        print(f"Final evaluation on test set for {dataset} with N={args.N}, r={args.r}")
        result = train_model(args, data=data, test=True)
        print(f"Final Result: {result}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'./results/test_results_{args.model_type}_adult.csv', 'a+') as f:
            f.write(f"{timestamp},{args.model_type},{dataset},{args.N},{args.r},{args.lin_dim},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")
