import numpy as np
from time import time
from functools import partial
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, root_mean_squared_error
from tensor.layers import TensorTrainLayer, TensorTrainLinearLayer, TensorNetworkLayer, CPDLayer
from tensor.network import SumOfNetworks
from tensor.bregman import SquareBregFunction

def root_mean_squared_error_torch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return root_mean_squared_error(y_true, y_pred)

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
        self.start_time = time()
        self.time_history = {}

    def convergence_criterion(self):
        elapsed_time = time() - self.start_time
        # Compute losses
        y_pred_val = self.model_predict(self.X_val)
        val_loss = self.loss_fn(self.y_val, y_pred_val)

        self.val_history[self.cur_degree] = val_loss
        self.time_history[self.cur_degree] = elapsed_time

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
                 lr=1.0, eps_start=1e-12, eps_end=1e-12,
                 batch_size=512, method='ridge_cholesky',
                 num_swipes=5,
                 model_type='tt',
                 verbose=0):
        self.N = N
        self.r = r
        self.output_dim = output_dim
        self.linear_dim = linear_dim if linear_dim is not None and linear_dim > 0 else None
        self.constrict_bond = constrict_bond
        self.perturb = perturb
        self.seed = seed
        self.device = device
        self.bf = bf if bf is not None else SquareBregFunction()
        self.lr = lr
        if num_swipes > 1:
            self.epss = np.geomspace(eps_start, eps_end, 2 * num_swipes).tolist() if eps_end != eps_start else [eps_end] * (2 * num_swipes)
        else:
            self.epss = np.geomspace(eps_start, eps_end, N).tolist()
        self.batch_size = batch_size
        self.method = method
        self.num_swipes = num_swipes
        self.model_type = model_type
        self.verbose = verbose

        self._model = None
        self.trajectory = []
        if self.perturb and self.output_dim > 1:
            raise ValueError("perturb not supported for output dim > 1")

    def _initialize_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set")
        if self.model_type == 'cpd':
            self._model = CPDLayer(self.N, self.r, self.input_dim,
                                   output_shape=self.output_dim,
                                   perturb=self.perturb,
                                   seed=self.seed).to(self.device)
        elif self.model_type.startswith('tt_type1'):
            if self.linear_dim is None or self.linear_dim >= self.input_dim:
                train_layers = [TensorTrainLayer(
                                    i,
                                    bond_dim=self.r,
                                    input_features=self.input_dim-1 if 'bias_first' in self.model_type and i != 1 else self.input_dim,
                                    output_shape=self.output_dim,
                                    constrict_bond=self.constrict_bond,
                                    perturb=self.perturb,
                                    seed=self.seed + i
                                ).tensor_network for i in range(1, self.N+1)]
                self._model = TensorNetworkLayer(SumOfNetworks(train_layers, only_bias_first='bias_first' in self.model_type, output_labels=train_layers[0].output_labels, train_linear=not ('_no_train_linear' in self.model_type))).to(self.device)
            else:
                train_layers = [TensorTrainLinearLayer(
                                i,
                                bond_dim=self.r,
                                input_features=self.input_dim-1 if 'bias_first' in self.model_type and i != 1 else self.input_dim,
                                linear_dim=self.linear_dim,
                                output_shape=self.output_dim,
                                constrict_bond=self.constrict_bond,
                                perturb=self.perturb,
                                seed=self.seed + i
                            ).tensor_network for i in range(1, self.N+1)]
                self._model = TensorNetworkLayer(SumOfNetworks(train_layers, only_bias_first='bias_first' in self.model_type, output_labels=train_layers[0].output_labels, train_linear=not ('_no_train_linear' in self.model_type))).to(self.device)

        elif self.linear_dim is None or self.linear_dim >= self.input_dim:
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
        if self.verbose > 2:
            print("Number of parameters:", self._model.num_parameters())

    def fit(self, X, y, X_val=None, y_val=None, validation_split=0.1, split_train=True):
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
        
        if self.verbose > 0:
            print("Number of parameters:", self._model.num_parameters())

        # Validation split if not provided
        if X_val is None or y_val is None:
            if split_train:
                n = X.shape[0]
                idx = np.arange(n)
                rng = np.random.RandomState(self.seed)
                rng.shuffle(idx)
                split = int(n * (1 - validation_split))
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

            if X_val.shape[1] != X_train.shape[1]:
                X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), dtype=torch.float64, device=self.device)), dim=1)

        # define convergence criterion function for progress printing
        epoch = 0
        self.trajectory = []
        def convergence_criterion():
            nonlocal epoch
            epoch += 1
            log_dict = {'epoch': epoch}
            y_pred_val = self._model.tensor_network.forward_batch(X_val, self.batch_size)
            rmse = root_mean_squared_error_torch(y_pred_val, y_val) #torch.sqrt(torch.mean((y_pred_val - y_val)**2)).item()
            log_dict['val_rmse'] = rmse
            # If more than 1 output dim, also print accuracy
            if y_val.shape[1] > 1:
                y_pred_labels = torch.argmax(y_pred_val, dim=1)
                y_true_labels = torch.argmax(y_val, dim=1)
                accuracy = (y_pred_labels == y_true_labels).float().mean().item()
                log_dict['val_accuracy'] = accuracy
            if self.verbose > 0:
                print(", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_dict.items()]))
            self.trajectory.append(log_dict)
            return False

        # Call accumulating_swipe
        self._model.tensor_network.accumulating_swipe(
            X_train, y_train, self.bf,
            batch_size=self.batch_size,
            lr=self.lr,
            eps=self.epss,
            convergence_criterion=convergence_criterion,
            orthonormalize=False,
            method=self.method,
            verbose=self.verbose,
            num_swipes=self.num_swipes,
            skip_second=False,
            direction='l2r',
            disable_tqdm=self.verbose < 3,
            eps_per_node=(self.num_swipes == 1) and (len(self.epss) == self.N)
        )

        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model.tensor_network.forward_batch(X, self.batch_size)
        return y_pred.detach().cpu().numpy()

    def score(self, X, y_true):
        # Return R2 score on X, y_true
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.cpu().numpy()
        # Append 1 to X for bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=self.device)), dim=1)
        y_pred = self._model.tensor_network.forward_batch(X, self.batch_size).squeeze().detach().cpu().numpy()
        return r2_score(y_true, y_pred)

def mirrored_cycle(seq, one_cycle=False):
    if not seq:
        return

    if one_cycle:
        yield from (list(seq) + list(reversed(seq[:-1])))
        return

    # forward part
    forward = list(seq)
    # backward part without repeating the endpoints
    backward = forward[-2:0:-1]  # excludes last and first
    pattern = forward + backward
    
    while True:
        for item in pattern:
            yield item

class TensorTrainBatchRegressor(TensorTrainRegressor):
    def __init__(self, *args, batch_size=1024, swipe_method='batch_unique', **kwargs):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.swipe_method = swipe_method
    
    def fit(self, X, y, X_val=None, y_val=None, validation_split=0.1, split_train=True):
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
        
        if self.verbose > 0:
            print("Number of parameters:", self._model.num_parameters())
        
        # Validation split if not provided
        if X_val is None or y_val is None:
            if split_train:
                n = X.shape[0]
                idx = np.arange(n)
                rng = np.random.RandomState(self.seed)
                rng.shuffle(idx)
                split = int(n * (1 - validation_split))
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
            if X_val.shape[1] != X_train.shape[1]:
                X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), dtype=torch.float64, device=self.device)), dim=1)
        
        # Calculate number of batches per epoch
        n_train = X_train.shape[0]
        n_batches = (n_train + self.batch_size - 1) // self.batch_size
        
        # Training loop with random batching
        epoch = 0
        self.trajectory = []
        
        # Create random number generator for batch sampling
        batch_rng = np.random.RandomState(self.seed)
        
        def convergence_criterion(batch_num):
            nonlocal epoch
            
            # Only run convergence check on the last batch of each epoch
            if batch_num % n_batches == 0:
                epoch += 1
                log_dict = {'epoch': epoch}
                y_pred_val = self._model.tensor_network.forward_batch(X_val, self.batch_size)
                rmse = root_mean_squared_error_torch(y_pred_val, y_val)
                log_dict['val_rmse'] = rmse
                
                # If more than 1 output dim, also print accuracy
                if y_val.shape[1] > 1:
                    y_pred_labels = torch.argmax(y_pred_val, dim=1)
                    y_true_labels = torch.argmax(y_val, dim=1)
                    accuracy = (y_pred_labels == y_true_labels).float().mean().item()
                    log_dict['val_accuracy'] = accuracy
                
                if self.verbose > 0:
                    print(", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_dict.items()]))
                
                self.trajectory.append(log_dict)
            
            return False
        
        # Training loop
        total_batches = self.num_swipes * n_batches
        batch_counter = 0
        
        for swipe in range(self.num_swipes):
            # Create random indices for this epoch
            indices = batch_rng.permutation(n_train)

            if self.swipe_method == 'batch_unique':
                block_seq = self._model.tensor_network.train_nodes
                block_iter = mirrored_cycle(block_seq, one_cycle=False)
                for batch_start in range(0, n_train, self.batch_size):
                    batch_counter += 1
                    batch_end = min(batch_start + self.batch_size, n_train)
                    batch_indices = indices[batch_start:batch_end]
                    
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    current_block = next(block_iter)

                    # Run single swipe on this batch
                    self._model.tensor_network.accumulating_swipe(
                        X_batch, y_batch, self.bf,
                        node_order=[current_block],
                        batch_size=-1,
                        lr=self.lr,
                        eps=self.epss,
                        convergence_criterion=lambda: convergence_criterion(batch_counter),
                        orthonormalize=False,
                        method=self.method,
                        verbose=self.verbose,  # Only verbose on last batch of epoch
                        num_swipes=1,  # Always 1 swipe per batch
                        skip_second=False,
                        direction='l2r',
                        disable_tqdm=self.verbose < 3,
                        eps_per_node=len(self.epss) == self.N
                    )
            elif self.swipe_method == 'batch_same':
                for batch_start in range(0, n_train, self.batch_size):
                    batch_counter += 1
                    batch_end = min(batch_start + self.batch_size, n_train)
                    batch_indices = indices[batch_start:batch_end]
                    
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Run single swipe on this batch
                    self._model.tensor_network.accumulating_swipe(
                        X_batch, y_batch, self.bf,
                        batch_size=-1,
                        lr=self.lr,
                        eps=self.epss,
                        orthonormalize=False,
                        method=self.method,
                        verbose=self.verbose,  # Only verbose on last batch of epoch
                        num_swipes=self.num_swipes,
                        skip_second=False,
                        direction='l2r',
                        disable_tqdm=self.verbose < 3,
                        eps_per_node=len(self.epss) == self.N
                    )
                    epoch += 1
                    log_dict = {'epoch': epoch}
                    y_pred_val = self._model.tensor_network.forward_batch(X_val, self.batch_size)
                    rmse = root_mean_squared_error_torch(y_pred_val, y_val)
                    log_dict['val_rmse'] = rmse

                    # If more than 1 output dim, also print accuracy
                    if y_val.shape[1] > 1:
                        y_pred_labels = torch.argmax(y_pred_val, dim=1)
                        y_true_labels = torch.argmax(y_val, dim=1)
                        accuracy = (y_pred_labels == y_true_labels).float().mean().item()
                        log_dict['val_accuracy'] = accuracy

                    if self.verbose > 0:
                        print(", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_dict.items()]))

                    self.trajectory.append(log_dict)
            elif self.swipe_method == 'batch_block':
                block_seq = self._model.tensor_network.train_nodes
                block_iter = mirrored_cycle(block_seq, one_cycle=True)
                for block in block_iter:
                    for batch_start in range(0, n_train, self.batch_size):
                        batch_counter += 1
                        batch_end = min(batch_start + self.batch_size, n_train)
                        batch_indices = indices[batch_start:batch_end]
                        
                        X_batch = X_train[batch_indices]
                        y_batch = y_train[batch_indices]

                        # Run single swipe on this batch
                        self._model.tensor_network.accumulating_swipe(
                            X_batch, y_batch, self.bf,
                            node_order=[block],
                            batch_size=-1,
                            lr=self.lr,
                            eps=self.epss,
                            convergence_criterion=lambda: convergence_criterion(batch_counter),
                            orthonormalize=False,
                            method=self.method,
                            verbose=self.verbose,  # Only verbose on last batch of epoch
                            num_swipes=1,  # Always 1 swipe per batch
                            skip_second=False,
                            direction='l2r',
                            disable_tqdm=self.verbose < 3,
                            eps_per_node=len(self.epss) == self.N
                        )
        
        return self

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
            model_predict=partial(self._model.tensor_network.forward_batch, batch_size=self.batch_size),
            get_model_weights=lambda: self._model.node_states(),
            loss_fn=root_mean_squared_error_torch,
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
            lr=self.lr,
            orthonormalize=False,
            verbose=self.verbose,
            num_swipes=1,      # always 1
            direction='l2r',
            disable_tqdm=self.verbose < 3,
            eps_per_node=True
        )

        # Get the best summary
        best_summary = self._early_stopping.best_summary()
        best_state_dict = best_summary['best_state_dict']
        self._best_degree = best_summary['best_degree']

        self._singular = not converged

        # Restore best state if available
        if best_state_dict is not None:
            self._model.load_node_states(best_state_dict, set_value=True)

        # # Now train for a couple of swipes across the best degree blocks
        # self._model.tensor_network.accumulating_swipe(
        #     X_train, y_train, self.bf,
        #     node_order=self._model.tensor_network.train_nodes[:self._best_degree],
        #     batch_size=self.batch_size,
        #     convergence_criterion=lambda: False,
        #     eps=np.geomspace(1.0, 1e-12, 2*5).tolist(),
        #     method=self.method,
        #     skip_second=False,
        #     eps_r=self.eps_r,
        #     lr=self.lr,
        #     orthonormalize=False,
        #     verbose=self.verbose == 1,
        #     num_swipes=5,
        #     direction='l2r',
        #     disable_tqdm=True
        # )

        return self