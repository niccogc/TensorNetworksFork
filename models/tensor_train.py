import torch
from tensor.layers import TensorTrainLayer, TensorOperatorLayer, TensorTrainLinearLayer
from tensor.bregman import XEAutogradBregman, SquareBregFunction
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_model(model, X, y_true, task):
    y_pred = model.predict(X)
    if task == 'classification':
        if y_true.ndim == 2:
            y_true = y_true.argmax(-1)
        acc = accuracy_score(y_true.cpu().numpy(), y_pred)
        return acc
    else:
        if y_true.ndim == 2:
            y_true = y_true.squeeze(-1)
        rmse = np.sqrt(mean_squared_error(y_true.cpu().numpy(), y_pred))
        return rmse

class TensorTrainWrapper:
    def __init__(self, input_dim, output_dim, tt_params, task='classification', device='cuda'):
        self.task = task
        self.device = device
        self.layer_type = tt_params.get('layer_type', 'tt')
        self.N = tt_params.get('N', 3)
        self.r = tt_params.get('r', 3)
        self.num_swipes = tt_params.get('num_swipes', 1)
        self.lr = tt_params.get('lr', 1.0)
        self.method = tt_params.get('method', 'exact')
        self.verbose = tt_params.get('verbose', 2)
        self.eps_min = tt_params.get('eps_min', 0.5)
        self.eps_max = tt_params.get('eps_max', 1.0)
        if self.eps_max == 0.0:
            self.eps = [0.0] * (2 * self.num_swipes)
        elif self.eps_min == 0.0:
            self.eps = np.geomspace(self.eps_max, 1e-12, num=2*self.num_swipes).tolist()
            self.eps[-1] = 0.0
        else:
            self.eps = np.geomspace(self.eps_max, self.eps_min, num=2*self.num_swipes).tolist()
        self.delta = tt_params.get('delta', 1.0)
        self.CB = tt_params.get('CB', -1)
        self.lin_bond = tt_params.get('lin_bond', 1)
        self.lin_dim = tt_params.get('lin_dim', 1)
        self.orthonormalize = tt_params.get('orthonormalize', False)
        self.timeout = tt_params.get('timeout', None)
        self.batch_size = tt_params.get('batch_size', 512)
        self.disable_tqdm = tt_params.get('disable_tqdm', False)
        self.early_stopping = tt_params.get('early_stopping', 20)
        self.track_eval = tt_params.get('track_eval', False)
        self.save_every = tt_params.get('save_every', 10)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(output_dim, int) and self.task == 'classification':
            self.output_dim = output_dim - 1
        self.output_shape = (output_dim,) if isinstance(output_dim, int) else output_dim
        if self.layer_type == 'tt':
            self.model = TensorTrainLayer(
                num_carriages=self.N,
                bond_dim=self.r,
                input_features=self.input_dim,
                output_shape=self.output_shape
            ).to(self.device)
        elif self.layer_type == 'operator':
            # Operator layer (optional, not default)
            ops = []
            p = self.input_dim
            N = self.N
            for n in range(N):
                H = torch.triu(torch.ones((1 if n == 0 else p, p), dtype=torch.get_default_dtype(), device=self.device), diagonal=0)
                D = torch.zeros((p, p, p, 1 if n == N-1 else p), dtype=torch.get_default_dtype(), device=self.device)
                for i in range(p):
                    D[i, i, i, 0 if n == N-1 else i] = 1
                C = torch.einsum('ij,j...->i...', H, D)
                ops.append(C)
            self.model = TensorOperatorLayer(
                operator=ops,
                input_features=self.input_dim,
                bond_dim=self.r,
                num_carriages=self.N,
                output_shape=self.output_shape
            ).to(self.device)
        elif self.layer_type == 'linear':
            self.model = TensorTrainLinearLayer(
                num_carriages=self.N,
                bond_dim=self.r,
                input_features=self.input_dim,
                linear_dim=int(self.lin_dim * self.input_dim) if 0.0 < self.lin_dim < 1.0 else int(self.lin_dim),
                linear_bond=self.lin_bond,
                output_shape=self.output_shape,
                connect_linear=self.lin_bond > 0,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

    def fit(self, X, y, X_val=None, y_val=None):
        X = X.to(self.device)
        y = y.to(self.device)
        # Bregman function
        if self.task == 'classification':
            with torch.inference_mode():
                y_pred = self.model(X[:64].to(self.device))
                w = 1/y_pred.std().item() if y_pred.std().item() > 0 else 1.0
            bf = XEAutogradBregman(w=w)
        else:
            bf = SquareBregFunction()
        if X_val is None and y_val is None:
            val_results = {}
            convergence_criterion = None
        else:
            val_results = {
                'best_metric': 0.0 if self.task == 'classification' else float('inf'),
                'best_state_dict': None,
                'early_stopping_count': 0,
                'count': 0,
            }
            def convergence_criterion(_, __):
                nonlocal val_results, X_val, y_val
                val_results['count'] += 1
                if val_results['count'] % self.save_every == 0:
                    with torch.inference_mode():
                        acc = evaluate_model(self, X_val, y_val, self.task)
                        if self.task == 'classification' and acc > val_results['best_metric'] or self.task == 'regression' and acc < val_results['best_metric']:
                            val_results['best_metric'] = acc
                            val_results['best_state_dict'] = self.model.node_states()
                            val_results['early_stopping_count'] = 0
                        else:
                            val_results['early_stopping_count'] += 1
                        if val_results['early_stopping_count'] > self.early_stopping and self.early_stopping > 0:
                            return True
                return False
        try:
            result = self.model.tensor_network.accumulating_swipe(
                X, y, bf,
                batch_size=self.batch_size,
                num_swipes=self.num_swipes,
                method=self.method,
                lr=self.lr,
                eps=self.eps,
                delta=self.delta,
                orthonormalize=self.orthonormalize,
                convergence_criterion=convergence_criterion,
                timeout=self.timeout,
                verbose=self.verbose,
                data_device=self.device,
                model_device=self.device,
                disable_tqdm=self.disable_tqdm
            )
        except Exception as e:
            print(f"Training failed: {e}")
            return False

        if 'best_state_dict' in val_results and val_results['best_state_dict'] is not None:
            self.model.load_node_states(val_results['best_state_dict'])
        return result
    def predict(self, X):
        X = X.to(self.device)
        with torch.inference_mode():
            self.model.tensor_network.reset_stacks()
            y_pred = self.model(X)
            if self.task == 'classification':
                if y_pred.shape[1] == 1:
                    y_pred = torch.cat([y_pred, torch.zeros_like(y_pred)], dim=1)
                return y_pred.argmax(dim=-1).cpu().numpy()
            else:
                return y_pred.cpu().numpy().squeeze()
