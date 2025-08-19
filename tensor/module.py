import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from tensor.layers import TensorTrainLayer, TensorTrainLinearLayer
from tensor.bregman import SquareBregFunction

def unexplained_variance(y_true, y_pred):
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=1, keepdim=True)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=1, keepdim=True)
    return (ss_res / ss_tot).mean().item()

class EarlyStopping:
    def __init__(self, X_train, y_train, X_val, y_val, model_predict, get_model_weights=None, loss_fn=None, abs_err=0.0, rel_err=0.0, early_stopping=5, verbose=0, start_degree=1):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_predict = model_predict
        self.get_model_weights = get_model_weights
        self.loss_fn = loss_fn
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.early_stop_count = 0
        self.cur_degree = start_degree
        self.best_degree = start_degree
        self.best_val_loss = np.inf
        self.best_train_loss = np.inf
        self.val_history = {}
        weights = self.get_model_weights()
        self.best_state_dict = weights if weights is not None else None

    def convergence_criterion(self):
        # Compute losses
        y_pred_val = self.model_predict(self.X_val)
        val_loss = self.loss_fn(self.y_val, y_pred_val)
        
        self.val_history[self.cur_degree] = val_loss

        train_loss = None
        if self.verbose > 0:
            y_pred_train = self.model_predict(self.X_train)
            train_loss = self.loss_fn(self.y_train, y_pred_train)
            print(f"Degree {self.cur_degree}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Measure improvement relative to previous best
        prev_best = self.best_val_loss
        improvement = prev_best - val_loss
        meets_abs = improvement >= self.abs_err
        meets_rel = improvement >= self.rel_err * abs(prev_best)
        meets_threshold = meets_abs or meets_rel

        if improvement > 0:
            # Always save the new best
            self.best_val_loss = val_loss
            if self.verbose > 0 and train_loss is not None:
                self.best_train_loss = train_loss
            self.best_degree = self.cur_degree
            if self.get_model_weights is not None:
                self.best_state_dict = self.get_model_weights()

            # Only reset patience if the improvement meets thresholds
            if meets_threshold:
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
        else:
            # No improvement at all
            self.early_stop_count += 1

        # Early stopping check
        if self.early_stop_count >= self.early_stopping:
            if self.verbose > 0:
                print(f"Converged degree: {self.best_degree} with best loss: {self.best_val_loss:.4f}")
            return True

        self.cur_degree += 1
        return False

    def best_summary(self):
        return {
            "best_degree": self.best_degree,
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "best_state_dict": self.best_state_dict,
        }

class TensorTrainRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, N=2, r=2, output_dim=1, linear_dim=None,  
                 constrict_bond=True, perturb=True, seed=42,
                 device='cuda', bf=None,
                 lr=1.0, eps_start=1e-12, eps_end=1e-12, eps_r=0.5, 
                 batch_size=512, method='ridge_cholesky',
                 num_swipes=5,
                 verbose=0):
        self.N = N
        self.r = r
        self.output_dim = output_dim
        self.linear_dim = linear_dim
        self.constrict_bond = constrict_bond
        self.perturb = perturb
        self.seed = seed
        self.device = device
        self.bf = bf if bf is not None else SquareBregFunction()
        self.lr = lr
        self.epss = np.geomspace(eps_start, eps_end, 2 * num_swipes).tolist() if eps_end != eps_start else [eps_end] * (2 * num_swipes)
        self.eps_r = eps_r
        self.batch_size = batch_size
        self.method = method
        self.num_swipes = num_swipes
        self.verbose = verbose
        
        self._model = None
        
    def _initialize_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set")
        if self.linear_dim is None or self.linear_dim >= self.input_dim:
            self._model = TensorTrainLayer(self.N, self.r, self.input_dim, 
                                           output_shape=self.output_dim,
                                           constrict_bond=self.constrict_bond,
                                           perturb=self.perturb,
                                           seed=self.seed).to(self.device)
        else:
            self._model = TensorTrainLinearLayer(self.N, self.r, self.input_dim, self.linear_dim,
                                                 output_shape=self.output_dim,
                                                 constrict_bond=self.constrict_bond,
                                                 perturb=self.perturb,
                                                 seed=self.seed).to(self.device)
    
    def fit(self, X, y):
        # X, y: numpy arrays or torch tensors
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones((X.shape[0], 1), dtype=torch.float64, device=self.device)), dim=1)
        
        if self._model is None:
            self.input_dim = X.shape[1]
            self._initialize_model()
        
        # define convergence criterion function for progress printing
        def convergence_criterion():
            y_pred_train = self._model(X)
            rmse = torch.sqrt(torch.mean((y_pred_train - y)**2))
            if self.verbose > 0:
                print('Train RMSE:', rmse.item())
            return False
        
        # Call accumulating_swipe
        self._model.tensor_network.accumulating_swipe(
            X, y, self.bf,
            batch_size=self.batch_size,
            lr=self.lr,
            eps=self.epss,
            eps_r=self.eps_r,
            convergence_criterion=convergence_criterion,
            orthonormalize=False,
            method=self.method,
            verbose=self.verbose,
            num_swipes=self.num_swipes,
            skip_second=False,
            direction='l2r',
            disable_tqdm=True
        )
        
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model(X)
        return y_pred.detach().cpu().numpy()
    
    def score(self, X, y_true):
        # Return R2 score on X, y_true
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.cpu().numpy()
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model(X).squeeze().detach().cpu().numpy()
        return r2_score(y_true, y_pred)

class TensorTrainRegressorEarlyStopping(TensorTrainRegressor):
    def __init__(self, *args, early_stopping=10, rel_err=1e-12, abs_err=1e-13, validation_split=0.1, split_train=False, **kwargs):
        # Warn the user first if these are set otherwise:
        if 'num_swipes' in kwargs and kwargs['num_swipes'] != 1:
            print("Warning: num_swipes is not set to 1 for early stopping. This setting will be overridden.")
        if 'perturb' in kwargs and not kwargs['perturb']:
            print("Warning: perturb is not set to True for early stopping. This setting will be overridden.")
        kwargs['num_swipes'] = 1
        kwargs['perturb'] = True
        super().__init__(*args, **kwargs)
        self.early_stopping = early_stopping
        self.rel_err = rel_err
        self.abs_err = abs_err
        self.validation_split = validation_split
        self.split_train = split_train

    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        # Validation split if not provided
        if X_val is None or y_val is None:
            if self.split_train:
                n = X.shape[0]
                idx = np.arange(n)
                rng = np.random.RandomState(self.seed)
                rng.shuffle(idx)
                split = int(n * (1 - self.validation_split))
                train_idx, val_idx = idx[:split], idx[split:]
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                X_train, y_train = X, y
                X_val, y_val = X, y
        else:
            if isinstance(X_val, np.ndarray):
                X_val = torch.tensor(X_val, dtype=torch.float64, device=self.device)
            if isinstance(y_val, np.ndarray):
                y_val = torch.tensor(y_val, dtype=torch.float64, device=self.device)
            if y_val.ndim == 1:
                y_val = y_val.unsqueeze(1)
            X_train, y_train = X, y

        # Append 1 to X for bias term
        X_train = torch.cat((X_train, torch.ones((X_train.shape[0], 1), dtype=torch.float64, device=self.device)), dim=1)
        X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), dtype=torch.float64, device=self.device)), dim=1)

        if self._model is None:
            self.input_dim = X_train.shape[1]
            self._initialize_model()

        self._early_stopping = EarlyStopping(
            X_train, y_train, X_val, y_val,
            model_predict=self._model,
            get_model_weights=lambda: self._model.node_states(),
            loss_fn=unexplained_variance,
            abs_err=self.abs_err,
            rel_err=self.rel_err,
            early_stopping=self.early_stopping,
            verbose=self.verbose
        )

        converged = self._model.tensor_network.accumulating_swipe(
            X_train, y_train, self.bf,
            batch_size=self.batch_size,
            convergence_criterion=self._early_stopping.convergence_criterion,
            eps=self.epss,
            method=self.method,
            skip_second=True,  # always True
            eps_r=self.eps_r,
            lr=self.lr,
            orthonormalize=False,
            verbose=self.verbose == 1,
            num_swipes=1,      # always 1
            direction='l2r',
            disable_tqdm=True
        )

        # Get the best summary
        best_summary = self._early_stopping.best_summary()
        best_state_dict = best_summary['best_state_dict']
        self._best_degree = best_summary['best_degree']

        self._singular = not converged


        # Restore best state if available
        if best_state_dict is not None:
            self._model.load_node_states(best_state_dict, set_value=True)

        return self