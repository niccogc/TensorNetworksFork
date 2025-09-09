from dotdict import DotDict
import torch
torch.set_default_dtype(torch.float64)
from train_grid_search import train_model
import pandas as pd
import numpy as np

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

    args.N = 3
    rs = [4, 8, 12]
    lin_dims = [0.25, 0.5, 0.75]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.75

    args.batch_size = 1024
    args.early_stopping = 10

    args.verbose = 1
    args.method = 'ridge_exact'

    seeds = list(range(42, 42+5))

    data = []
    for path, task in paths:
        dataset = path.split('/')[-1].replace('_tensor.pt', '')
        args.path = path
        args.task = task
        for r in rs:
            for lin_dim in lin_dims:
                for seed in seeds:
                    args.r = r
                    args.lin_dim = lin_dim
                    args.seed = seed
                    print(f"Training {dataset} with N={args.N}, r={r}, lin_dim={lin_dim}")
                    result = train_model(args)
                    data.append((dataset, args.N, r, lin_dim, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                    print(f"Result: {result}")
    
    df = pd.DataFrame(data, columns=['dataset', 'N', 'r', 'lin_dim', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'converged_epoch', 'seed'])
    df['num_swipes'] = args.num_swipes
    df['eps_start'] = args.eps_start
    df['eps_decay'] = args.eps_decay
    df['early_stopping'] = args.early_stopping
    df['model_type'] = args.model_type
    df.to_csv('./results/grid_search_results_tt_lin.csv', index=False)
