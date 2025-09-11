from dotdict import DotDict
import torch
torch.set_default_dtype(torch.float64)
from train_grid_search import train_model
import pandas as pd
import numpy as np

if __name__ == '__main__':
    args = DotDict()
    paths = [
        ('/work3/aveno/Tabular/data/processed/gesture_tensor.pt', 'classification'),
        ('/work3/aveno/Tabular/data/concrete_tensor.pt', 'regression'),
        ('/work3/aveno/Tabular/data/energyprediction_tensor.pt', 'regression'),
        ('/work3/aveno/Tabular/data/processed/higgs_small_tensor.pt', 'classification'),
    ]
    args.device = 'cuda'
    args.data_device = 'cuda'
    args.model_type = 'cpd'

    Ns = [3, 4, 5]
    rs = [16, 64, 144]
    args.num_swipes = 100
    args.lr = 1.0
    args.eps_start = 5.0
    args.eps_decay = 0.75

    args.batch_size = 1024
    args.early_stopping = 10

    args.verbose = 1
    args.method = 'ridge_cholesky'
    args.lin_dim = None

    seeds = list(range(42, 42+5))

    data = []
    for path, task in paths:
        dataset = path.split('/')[-1].replace('_tensor.pt', '')
        args.path = path
        args.task = task
        for N in Ns:
            for r in rs:
                args.N = N
                args.r = r
                for seed in seeds:
                    args.seed = seed
                    print(f"Training {dataset} with N={N}, r={r}")
                    result = train_model(args)
                    data.append((dataset, N, r, np.nan, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], result['converged_epoch'], seed))
                    print(f"Result: {result}")
    
    df = pd.DataFrame(data, columns=['dataset', 'N', 'r', 'lin_dim', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'converged_epoch', 'seed'])
    df['num_swipes'] = args.num_swipes
    df['eps_start'] = args.eps_start
    df['eps_decay'] = args.eps_decay
    df['early_stopping'] = args.early_stopping
    df['model_type'] = args.model_type
    df.to_csv('./results/grid_search_results_cpd.csv', index=False)
