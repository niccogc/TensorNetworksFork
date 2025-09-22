import sys
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from load_ucirepo import get_ucidata
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
from datetime import datetime
from dotdict import DotDict
import pandas as pd
import xgboost as xgb

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

def error_rate_torch(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

datasets = [
  # ('iris', 53, 'classification'),
  # ('adult', 2, 'classification'),
  # ('hearth', 45, 'classification'),
  # ('winequalityc', 186, 'classification'),
  # ('breast', 17, 'classification'),
  # ('bank', 222, 'classification'),
  # ('wine', 109, 'classification'),
  # ('car_evaluation', 19, 'classification'),
  # ('student_dropout', 697, 'classification'),
  # ('mushrooms', 73, 'classification'),
  # ('student_perf', 320, 'regression'),
  # ('abalone', 1, 'regression'),
  # ('obesity', 544, 'regression'),
  # ('bike', 275, 'regression'),
  # ('realstate', 477, 'regression'),
  # ('energy_efficiency', 242, 'regression'),
  # ('concrete', 165, 'regression'),
  # ('ai4i', 601, 'regression'),
  # ('appliances', 374, 'regression'),
  # ('popularity', 332, 'regression'),
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

class XGBoost:
    """
    A unified wrapper for XGBoost that handles both regression and classification tasks.
    """
    def __init__(self, task='regression', xgb_params=None):
        self.task = task
        if xgb_params is None:
            xgb_params = {}

        # Use GPU for XGBoost if available and device is 'cuda'
        if torch.cuda.is_available() and xgb_params.get('device') == 'cuda':
             xgb_params['tree_method'] = 'hist'
        
        if self.task == 'regression':
            self.model = xgb.XGBRegressor(**xgb_params)
        elif self.task == 'classification':
            self.model = xgb.XGBClassifier(eval_metric='mlogloss', **xgb_params)
        else:
            raise ValueError(f"Unknown task: {task}")

    def fit(self, X, y):
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        if self.task == 'classification' and y_np.ndim == 2:
            y_np = y_np.argmax(axis=-1)

        self.model.fit(X_np, y_np)

    def predict(self, X):
        X_np = X.cpu().numpy()
        if self.task == 'classification':
            return self.model.predict_proba(X_np)
        else:
            return self.model.predict(X_np)

def train_model(args, data=None, test=False):
    if data is None:
        data = get_ucidata(args.dataset_id, args.task, device=args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Model setup
    xgb_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.lr,
        'device': args.device  # Pass device to XGBoost
    }
    model = XGBoost(task=args.task, xgb_params=xgb_params)
    model.fit(X_train, y_train)
    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = np.nan
    converged_epoch = np.nan

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
    args = DotDict()
    args.device = 'cuda'
    args.data_device = 'cuda'

    args.model_type = 'xgboost'
    args.lr = 0.1  # fixed learning rate

    # Hyperparameter grids
    n_estimatorss = [50, 100, 200, 500]
    max_depths = [3, 6, 9]

    seeds = list(range(42, 42 + 5))

    for dataset, dataset_id, task in datasets:
        results = []
        data = get_ucidata(dataset_id, task, args.data_device)

        args.task = task

        for n_estimators in n_estimatorss:
            for max_depth in max_depths:
                try:
                    args.n_estimators = n_estimators
                    args.max_depth = max_depth

                    print(f"Training {dataset} with n_estimators {n_estimators}, max_depth {max_depth}", file=sys.stdout, flush=True)
                    result = train_model(args, data=data, test=False)

                    results.append((
                        dataset,
                        n_estimators,
                        max_depth,
                        result['val_rmse'],
                        result['val_r2'],
                        result['val_accuracy'],
                        result['num_params']
                    ))

                    print(f"Result: {result}", file=sys.stdout, flush=True)
                except KeyboardInterrupt:
                    print("Interrupted by user, exiting...", file=sys.stdout, flush=True)
                    exit(0)
                # except:
                #     print("Failed, skipping...", file=sys.stdout, flush=True)
                #     torch.cuda.empty_cache()
                #     continue

        # Build per-dataset results frame
        df = pd.DataFrame(
            results,
            columns=['dataset', 'n_estimators', 'max_depth', 'val_rmse', 'val_r2', 'val_accuracy', 'num_params']
        )
        df['model_type'] = args.model_type
        df['learning_rate'] = args.lr

        if len(df) == 0:
            exit(0)

        df.to_csv(f'./results/{dataset}_ablation_results_{args.model_type}.csv', index=False)

        # Aggregate across seeds
        group_by_cols = ['n_estimators', 'max_depth']
        df_agg = df.groupby(group_by_cols).agg({'val_rmse': 'mean', 'val_accuracy': 'mean'}).reset_index()

        if task == 'regression':
            best_row = df_agg.loc[df_agg['val_rmse'].idxmin()]
        else:
            best_row = df_agg.loc[df_agg['val_accuracy'].idxmax()]

        # Reconstruct best setting and evaluate on test set
        args.n_estimators = int(best_row['n_estimators'])
        args.max_depth = int(best_row['max_depth'])

        print(f"Final evaluation on test set for {dataset}", file=sys.stdout, flush=True)
        result = train_model(args, data=data, test=True)
        print(f"Final Result: {result}", file=sys.stdout, flush=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'./results/test_results_{args.model_type}.csv', 'a+') as f:
            f.write(
                f"{timestamp},{args.model_type},{dataset},{args.n_estimators},"
                f"{args.max_depth},{result['test_rmse']},{result['test_r2']},"
                f"{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n"
            )
