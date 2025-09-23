import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score

class ConvNet(nn.Module):
    """A configurable CNN for MNIST classification."""

    def __init__(self, num_classes=10, dropout_rate=0.5, base_channels=32, num_conv_blocks=3, fc_hidden_dims=None):
        super(ConvNet, self).__init__()

        self.num_conv_blocks = num_conv_blocks

        # Default FC hidden dimensions if not specified
        if fc_hidden_dims is None:
            fc_hidden_dims = [256, 128]

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        in_channels = 1
        current_channels = base_channels

        for i in range(num_conv_blocks):
            self.conv_blocks.append(nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1))
            self.bn_blocks.append(nn.BatchNorm2d(current_channels))
            in_channels = current_channels
            current_channels *= 2  # Double channels each block

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after conv blocks for FC layers
        # MNIST is 28x28, after num_conv_blocks pooling operations: 28 / (2^num_conv_blocks)
        final_size = 28 // (2 ** num_conv_blocks)
        if final_size < 1:
            final_size = 1  # Minimum size

        # Final channels from last conv block
        final_channels = base_channels * (2 ** (num_conv_blocks - 1))
        flattened_size = final_channels * final_size * final_size

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()

        # Input layer
        if len(fc_hidden_dims) > 0:
            self.fc_layers.append(nn.Linear(flattened_size, fc_hidden_dims[0]))

            # Hidden layers
            for i in range(len(fc_hidden_dims) - 1):
                self.fc_layers.append(nn.Linear(fc_hidden_dims[i], fc_hidden_dims[i + 1]))

            # Output layer
            self.fc_layers.append(nn.Linear(fc_hidden_dims[-1], num_classes))
        else:
            # Direct connection to output if no hidden layers
            self.fc_layers.append(nn.Linear(flattened_size, num_classes))

    def forward(self, x):
        # Convolutional blocks
        for i in range(self.num_conv_blocks):
            x = self.conv_blocks[i](x)
            x = F.relu(self.bn_blocks[i](x))
            x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout (except last layer)
        for i, fc_layer in enumerate(self.fc_layers[:-1]):
            x = F.relu(fc_layer(x))
            x = self.dropout(x)

        # Final output layer (no dropout)
        x = self.fc_layers[-1](x)

        return x


def error_rate_torch(y_true, y_pred):
    """Calculates the error rate (1 - accuracy) from torch tensors."""
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True, verbose=1):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_model(args):
    """
    Trains and evaluates the CNN model based on provided arguments.

    Args:
        args: An argparse.Namespace object containing all hyperparameters.

    Returns:
        A dictionary containing the results of the run.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose > 0:
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # --- 1. Data Loading and Preprocessing ---
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.dataset == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=args.data_path, train=True, transform=transform_train, download=True)
        test_dataset = torchvision.datasets.MNIST(root=args.data_path, train=False, transform=transform_test, download=True)
    elif args.dataset == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=args.data_path, train=True, transform=transform_train, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=args.data_path, train=False, transform=transform_test, download=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create validation split
    train_size = int((1 - args.validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 2. Model Definition ---
    # Parse FC hidden dimensions if provided
    fc_hidden_dims = None
    if args.fc_hidden_dims:
        fc_hidden_dims = [int(x) for x in args.fc_hidden_dims.split(',')]

    model = ConvNet(
        num_classes=10,
        dropout_rate=args.dropout_rate,
        base_channels=args.base_channels,
        num_conv_blocks=args.num_conv_blocks,
        fc_hidden_dims=fc_hidden_dims
    ).to(device)

    if args.verbose > 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total params: {total_params}, Trainable params: {trainable_params}')

    # --- 3. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=args.verbose)

    # --- 4. Training ---
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            if args.verbose >= 2 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if args.verbose > 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%')

        scheduler.step()

        # Early stopping check
        if early_stopping(avg_val_loss, model):
            break

    # --- 5. Evaluation ---
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    test_accuracy = 100. * test_correct / test_total

    if args.verbose > 0:
        print(f"Test Accuracy: {test_accuracy:.2f}%")

    # --- 6. Return Results ---
    return {
        'dataset': args.dataset,
        'model_type': 'cnn',
        'base_channels': args.base_channels,
        'num_conv_blocks': args.num_conv_blocks,
        'fc_hidden_dims': args.fc_hidden_dims,
        'test_accuracy': test_accuracy,
        'num_epochs': epoch + 1,  # Actual number of epochs trained
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'early_stopping': args.early_stopping,
        'seed': args.seed,
        'validation_split': args.validation_split
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model on image classification.")

    # Model and Data Hyperparameters
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'], help='Dataset to use.')
    parser.add_argument('--data_path', type=str, default='/work3/aveno/MNIST/data')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for regularization.')

    # Network Architecture Parameters
    parser.add_argument('--base_channels', type=int, default=32, help='Base number of channels in first conv layer (doubles each block).')
    parser.add_argument('--num_conv_blocks', type=int, default=3, help='Number of convolutional blocks.')
    parser.add_argument('--fc_hidden_dims', type=str, default='256,128', help='Comma-separated list of FC hidden layer dimensions (e.g., "256,128").')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization.')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for learning rate scheduler.')
    parser.add_argument('--lr_gamma', type=float, default=0.7, help='Gamma for learning rate scheduler.')

    # Early Stopping Hyperparameters
    parser.add_argument('--early_stopping', type=int, default=10, help='Patience for early stopping.')

    # General Configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of training data to use for validation.')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=silent, 1=progress, 2=detailed).')

    args = parser.parse_args()

    if args.verbose > 0:
        print("Starting CNN training with the following configuration:")
        print(args)

    results_dict = train_model(args)

    # Format the accuracy to two decimal places for the final print
    results_dict['test_accuracy'] = f"{results_dict['test_accuracy']:.2f}"

    # Print the final results string as requested
    output_string = ",".join(map(str, results_dict.values()))
    print(output_string)