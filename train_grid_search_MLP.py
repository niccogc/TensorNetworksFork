import sys
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)
import numpy as np
from load_ucirepo import get_ucidata
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
from datetime import datetime
from dotdict import DotDict
import pandas as pd
from models.tensor_train import EarlyStopping

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return root_mean_squared_error(y_true, y_pred)

def error_rate_torch(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
    y_true_labels = y_true.cpu().detach().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

datasets = [
    ('iris', 53, 'classification'),
    ('adult', 2, 'classification'),
    ('hearth', 45, 'classification'),
    ('winequalityc', 186, 'classification'),
    ('breast', 17, 'classification'),
    ('bank', 222, 'classification'),
    ('wine', 109, 'classification'),
    ('car_evaluation', 19, 'classification'),
    ('student_dropout', 697, 'classification'),
    ('mushrooms', 73, 'classification'),
    ('student_perf', 320, 'regression'),
    ('abalone', 1, 'regression'),
    ('obesity', 544, 'regression'),
    ('bike', 275, 'regression'),
    ('realstate', 477, 'regression'),
    ('energy_efficiency', 242, 'regression'),
    ('concrete', 165, 'regression'),
    ('ai4i', 601, 'regression'),
    ('appliances', 374, 'regression'),
    ('popularity', 332, 'regression'),
    ('seoulBike', 560, 'regression'),
]


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

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) class.
    """
    def __init__(self, input_dim, output_dim, channels, task='regression'):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output.
            channels (list): A list of integers, where each integer is the
                                   number of neurons in a hidden layer.
            task (str): The task type, either 'regression' or 'classification'.
        """
        super(MLP, self).__init__()
        self.task = task
        
        layers = []
        prev_dim = input_dim
        for num_neurons in channels:
            layers.append(nn.Linear(prev_dim, num_neurons))
            layers.append(nn.LayerNorm(num_neurons))
            layers.append(nn.ReLU())
            prev_dim = num_neurons
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs the forward pass of the model.
        """
        return self._model(x)

    def predict(self, X):
        """
        Makes predictions on the given data.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            if self.task == 'classification':
                return torch.softmax(outputs, dim=-1).cpu().numpy()
            else:
                return outputs.cpu().numpy()

def train_model(args, data=None, test=False):
    if data is None:
        data = get_ucidata(args.dataset_id, args.task, device=args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Model setup
    input_dim = X_train.shape[1]
    if y_train.ndim == 1 and args.task == 'regression':
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    output_dim = y_train.shape[1] if args.task == 'regression' else len(torch.unique(y_train.to(dtype=torch.long)))
    
    model = MLP(input_dim, output_dim, args.channels, args.task).to(args.device)

    early_stopper = EarlyStopping(
        X_val, y_val,
        model_predict=model,
        get_model_weights=model.state_dict,
        loss_fn=root_mean_squared_error_torch if args.task == 'regression' else error_rate_torch,
        abs_err=args.abs_err,
        rel_err=args.rel_err,
        early_stopping=args.early_stopping,
        verbose=args.verbose
    )
    
    # Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() if args.task == 'regression' else nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], args.batch_size):
            indices = permutation[i:i+args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if args.task == 'regression':
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs, batch_y.long().squeeze())

            loss.backward()
            optimizer.step()
        if early_stopper.convergence_criterion():
            if args.verbose > 0:
                print(f"Early stopping at epoch {epoch+1}")
            break
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    converged_epoch = early_stopper.epoch

    report_dict = {}
    if args.task == 'classification':
        report_dict['val_rmse'] = np.nan
        report_dict['val_r2'] = np.nan
        report_dict['val_accuracy'] = val_score
        report_dict['num_params'] = num_params
        report_dict['converged_epoch'] = converged_epoch
    else:
        # Calculate R2 score as well
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        report_dict['val_rmse'] = val_score
        report_dict['val_r2'] = r2_val
        report_dict['val_accuracy'] = np.nan
        report_dict['num_params'] = num_params
        report_dict['converged_epoch'] = converged_epoch

    if test:
        test_score = evaluate_model(model, X_test, y_test, metric)
        if args.task == 'classification':
            report_dict['test_rmse'] = np.nan
            report_dict['test_r2'] = np.nan
            report_dict['test_accuracy'] = test_score
        else:
            r2_test = evaluate_model(model, X_test, y_test, metric='r2')
            report_dict['test_rmse'] = test_score
            report_dict['test_r2'] = r2_test
            report_dict['test_accuracy'] = np.nan
    return report_dict

if __name__ == '__main__':
    skip_grid_search = True  # Set to True to skip grid search and load from CSV

    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'

    num_layerss = [1, 3, 5]
    num_channels = [16, 64, 256]
    args.batch_size = 256
    args.model_type = f'mlp'
    args.epochs = 100
    args.lr = 1e-3
    args.abs_err = 1e-4
    args.rel_err = 1e-4
    args.verbose = 1
    args.early_stopping = 20

    seeds = list(range(42, 42+5))
    for dataset, dataset_id, task in datasets:
        data = get_ucidata(dataset_id, task, args.data_device)
        num_features = data[0].shape[1]
        args.early_stopping = max(10, num_features+1)
        args.task = task

        if skip_grid_search:
            # Load existing results from CSV
            df = pd.read_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv')
            print(f"Loaded existing results for {dataset}", file=sys.stdout, flush=True)
        else:
            # Perform grid search
            results = []
            for num_layers in num_layerss:
                for num_channel in num_channels:
                    for seed in seeds:
                        try:
                            channels = [num_channel] * num_layers
                            args.channels = channels
                            args.seed = seed
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)

                            print(f"Training {dataset} with {num_layers} layers, {num_channel} channels, seed {seed}", file=sys.stdout, flush=True)
                            result = train_model(args, data=data, test=False)
                            results.append((dataset, num_channel, num_layers, result['val_rmse'], result['val_r2'], result['val_accuracy'], result['num_params'], seed))
                            print(f"Result: {result}", file=sys.stdout, flush=True)
                        except KeyboardInterrupt:
                            print("Interrupted by user, exiting...", file=sys.stdout, flush=True)
                            exit(0)
                        # except:
                        #     print("Failed, skipping...", file=sys.stdout, flush=True)
                        #     torch.cuda.empty_cache()
                        #     continue

            df = pd.DataFrame(results, columns=['dataset', 'num_channels', 'num_layers', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params', 'seed'])
            df['model_type'] = args.model_type
            df['early_stopping'] = args.early_stopping

            if len(df) == 0:
                exit(0)

            df.to_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv', index=False)

        # Take the best one and run it on the test set
        group_by_cols = ['num_channels', 'num_layers']
        df_agg = df.groupby(group_by_cols).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()

        if task == 'regression':
            best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
        else:
            best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]

        # Reconstruct best net and train it multiple times
        args.num_layers = int(best_row['num_layers'])
        args.num_channels = int(best_row['num_channels'])
        args.channels = [args.num_channels] * args.num_layers

        # Run 5 test runs with different seeds
        test_seeds = [1337, 2024, 3141, 4242, 5555]
        for test_seed in test_seeds:
            args.seed = test_seed
            torch.manual_seed(test_seed)
            torch.cuda.manual_seed_all(test_seed)
            print(f"Final evaluation on test set for {dataset} with seed {test_seed}", file=sys.stdout, flush=True)
            result = train_model(args, data=data, test=True)
            print(f"Final Result: {result}", file=sys.stdout, flush=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'./results/test_results_{args.model_type}.csv', 'a+') as f:
                f.write(f"{timestamp},{args.model_type},{dataset},{args.num_layers},{args.num_channels},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")
