import torch
from tensor.layers import TensorTrainLayer, TensorOperatorLayer
from tensor.bregman import XEAutogradBregman, SquareBregFunction
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_squared_error


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
        self.eps_min = tt_params.get('eps_min', 0.5)
        self.eps_max = tt_params.get('eps_max', 1.0)
        self.eps = np.geomspace(self.eps_max, self.eps_min, num=2*self.num_swipes).tolist()
        self.delta = tt_params.get('delta', 1.0)
        self.orthonormalize = tt_params.get('orthonormalize', False)
        self.timeout = tt_params.get('timeout', None)
        self.batch_size = tt_params.get('batch_size', 512)
        self.disable_tqdm = tt_params.get('disable_tqdm', False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_shape = (output_dim,) if isinstance(output_dim, int) else output_dim
        if self.layer_type == 'tt':
            self.model = TensorTrainLayer(
                num_carriages=self.N,
                bond_dim=self.r,
                input_features=self.input_dim,
                output_shape=self.output_shape
            ).to(self.device)
        else:
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
    def fit(self, X, y, X_val=None, y_val=None):
        # Bregman function
        if self.task == 'classification':
            with torch.inference_mode():
                y_pred = self.model(X[:64].to(self.device))
                w = 1/y_pred.std().item() if y_pred.std().item() > 0 else 1.0
            bf = XEAutogradBregman(w=w)
        else:
            bf = SquareBregFunction()
        def convergence_criterion(_, __):
            return False  # No early stopping for now
        try:
            return self.model.tensor_network.accumulating_swipe(
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
                verbose=2,
                data_device=self.device,
                model_device=self.device,
                disable_tqdm=self.disable_tqdm
            )
        except Exception as e:
            print('Training failed:', e)
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X.to(self.device))
            if self.task == 'classification':
                # Pad if needed (for binary)
                if y_pred.shape[1] == 1:
                    y_pred = torch.cat([y_pred, torch.zeros_like(y_pred)], dim=1)
                return y_pred.argmax(dim=-1).cpu().numpy()
            else:
                return y_pred.cpu().numpy().squeeze()
