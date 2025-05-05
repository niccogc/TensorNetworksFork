import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], activation='relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act = nn.ReLU if activation == 'relu' else nn.Tanh
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class MLPWrapper:
    def __init__(self, input_dim, output_dim, mlp_params=None, task='regression'):
        if mlp_params is None:
            mlp_params = {}
        hidden_layers = mlp_params.get('hidden_layers', [64, 64])
        activation = mlp_params.get('activation', 'relu')
        self.lr = mlp_params.get('lr', 1e-1)
        self.epochs = mlp_params.get('epochs', 50)
        self.batch_size = mlp_params.get('batch_size', 128)
        self.device = mlp_params.get('device', 'cuda')
        self.task = task
        self.model = MLP(input_dim, output_dim, hidden_layers, activation).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
    def fit(self, X, y):
        X = X.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long if self.task == 'classification' else torch.float32)
        # Convert one-hot to class labels if needed
        if self.task == 'classification' and y.ndim == 2:
            y = y.argmax(-1)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        loss_mean = np.inf
        for _ in (t_bar:=tqdm(range(self.epochs))):
            losses = []
            for xb, yb in loader:
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                loss_batch = loss.item()
                losses.append(loss_batch)
                t_bar.set_postfix(loss=loss_mean, loss_batch=loss_batch)
            loss_mean = np.mean(losses)
    def predict(self, X):
        self.model.eval()
        X = X.to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(X)
            if self.task == 'classification':
                return out.argmax(dim=1).cpu().numpy()
            else:
                return out.squeeze(-1).cpu().numpy()
