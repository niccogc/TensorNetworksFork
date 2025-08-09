
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from tensor.layers import TensorTrainLayer, TensorTrainLinearLayer
from tensor.bregman import SquareBregFunction

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