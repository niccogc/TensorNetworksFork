import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score

# ---- Tabular data loader ----
def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_val = data['X_val'].to(device)
    y_val = data['y_val'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    if torch.all(X_train[:, -1] == 1) and torch.all(X_val[:, -1] == 1) and torch.all(X_test[:, -1] == 1):
        X_train = X_train[:, :-1]
        X_val = X_val[:, :-1]
        X_test = X_test[:, :-1]

    return X_train, y_train, X_val, y_val, X_test, y_test

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
    def __init__(self, input_dim, output_dim, hidden_neurons, task='regression'):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output.
            hidden_neurons (list): A list of integers, where each integer is the
                                   number of neurons in a hidden layer.
            task (str): The task type, either 'regression' or 'classification'.
        """
        super(MLP, self).__init__()
        self.task = task
        
        layers = []
        prev_dim = input_dim
        for num_neurons in hidden_neurons:
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

def train_model(args, data=None):
    if data is None:
        data = load_tabular_data(args.path, args.data_device)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Model setup
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if args.task == 'regression' else len(torch.unique(y_train.to(dtype=torch.long)))
    
    model = MLP(input_dim, output_dim, args.hidden_neurons, args.task).to(args.device)
    
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

    # Unified evaluation
    metric = 'accuracy' if args.task == 'classification' else 'rmse'
    val_score = evaluate_model(model, X_val, y_val, metric)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    converged_epoch = np.nan # Placeholder as there's no early stopping

    if args.task == 'classification':
        return {'val_rmse': np.nan, 'val_r2': np.nan, 'val_accuracy': val_score, 'num_params': num_params, 'converged_epoch': converged_epoch}
    else:
        # Calculate R2 score as well
        r2_val = evaluate_model(model, X_val, y_val, metric='r2')
        return {'val_rmse': val_score, 'val_r2': r2_val, 'val_accuracy': np.nan, 'num_params': num_params, 'converged_epoch': converged_epoch}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensor Network Training for Tabular Data')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for data files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (without .pt extension)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to store the dataset (cpu or cuda)')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Task type: regression or classification')

    # MLP hyperparameters
    parser.add_argument('--hidden_neurons', type=int, nargs='+', default=[128, 64], help='Number of neurons in hidden layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    args.path = os.path.join(args.data_dir, args.dataset_name + '_tensor.pt') # Corrected to standard .pt extension
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    result = train_model(args)
    print(result)