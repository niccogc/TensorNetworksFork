import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
import tempfile
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

# ---- Dataset registry ----
DATASETS = {
    'mnist': {
        'class': torchvision.datasets.MNIST,
        'root': '/work3/s183995/MNIST/data',
        'num_classes': 10,
        'input_channels': 1,
        'normalize': ((0.1307,), (0.3081,)),
        # Define kernel configs: index -> (size, stride, padding)
        # MNIST is 28x28, so kernel size 7 is a good choice
        'kernels': {
            1: (1, 1, 0),
            2: (2, 1, 0),
            3: (2, 1, 1),
            4: (2, 2, 0),
            5: (4, 2, 0),
            6: (4, 2, 2),
            7: (4, 4, 0),
            8: (7, 7, 0),
            9: (14, 7, 0),
            10: (14, 7, 7),
            11: (14, 14, 0),
        },
    },
    'cifar10': {
        'class': torchvision.datasets.CIFAR10,
        'root': '/work3/s183995/CIFAR10/data',
        'num_classes': 10,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Define kernel configs: index -> (size, stride, padding)
        # CIFAR10 is 32x32, so kernel size 8 is a good choice
        'kernels': {
            1: (1, 1, 0),
            2: (2, 1, 0),
            3: (2, 1, 1),
            4: (2, 2, 0),
            5: (4, 2, 0),
            6: (4, 2, 2),
            7: (4, 4, 0),
            8: (8, 4, 0),
            9: (8, 4, 4),
            10: (8, 8, 0),
            11: (16, 8, 0),
            12: (16, 8, 8),
            13: (16, 16, 0),
        },
    },
    'cifar100': {
        'class': torchvision.datasets.CIFAR100,
        'root': '/work3/s183995/CIFAR100/data',
        'num_classes': 100,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Define kernel configs: index -> (size, stride, padding)
        # CIFAR100 is 32x32, so kernel size 8 is a good choice
        'kernels': {
            1: (1, 1, 0),
            2: (2, 1, 0),
            3: (2, 1, 1),
            4: (2, 2, 0),
            5: (4, 2, 0),
            6: (4, 2, 2),
            7: (4, 4, 0),
            8: (8, 4, 0),
            9: (8, 4, 4),
            10: (8, 8, 0),
            11: (16, 8, 0),
            12: (16, 8, 8),
            13: (16, 16, 0),
        },
    },
    'imagenet': {
        'class': torchvision.datasets.ImageNet,
        'root': '/work3/s183995/ImageNet/data',
        'num_classes': 100,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Define kernel configs: index -> (size, stride, padding)
        'kernels': {
        },
    },
}

def get_data_loaders(dataset_name, batch_size, download, data_path=None):
    ds = DATASETS[dataset_name]
    root = data_path if data_path else ds['root']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*ds['normalize'])
    ])
    DatasetClass = ds['class']
    train_dataset = DatasetClass(root=root, train=True, transform=transform, download=download)
    test_dataset = DatasetClass(root=root, train=False, transform=transform, download=download)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, ds['num_classes']

def preprocess_batches(loader, num_classes, kernel_size, stride, device, padding=0):
    samples = []
    labels = []
    for images, lbls in loader:
        samples.append(images)
        labels.append(lbls)
    x = torch.cat(samples, dim=0)
    y = torch.cat(labels, dim=0)
    x = F.unfold(x, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding).transpose(-2, -1)
    x = torch.cat((x, torch.ones((x.shape[0], 1, x.shape[2]), device=x.device)), dim=-2).to(device)
    y = F.one_hot(y, num_classes=num_classes).to(dtype=torch.float64, device=device)
    return x, y

def train_with_timeout(args, kernel_size, stride, padding, N, r, CB, xinp_train, y_train, model_path):
    import torch
    import os
    from tensor.layers import TensorConvolutionTrainLayer
    from tensor.bregman import XEAutogradBregman
    from sklearn.metrics import balanced_accuracy_score
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    wandb_enabled = False
    if args.wandb_project:
        import wandb
        config = vars(args)
        config['kernel_size'] = kernel_size
        config['stride'] = stride
        config['padding'] = padding
        wandb.init(project=args.wandb_project, config=config, reinit=True)
        wandb_enabled = True
    layer = TensorConvolutionTrainLayer(
        num_carriages=N,
        bond_dim=r,
        num_patches=xinp_train.shape[1],
        patch_pixels=xinp_train.shape[2],
        output_shape=y_train.shape[1]-1,
        convolution_bond=CB
    ).to(device)
    print('Num params:', layer.num_parameters())
    with torch.inference_mode():
        y_pred = layer(xinp_train[:64])
        w = 1/y_pred.std().item()
        del y_pred
    bf = XEAutogradBregman(w=w)
    def convergence_criterion(y_pred, y_true):
        y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
        balanced_acc = balanced_accuracy_score(y_true.argmax(dim=-1).cpu().numpy(), y_pred.argmax(dim=-1).cpu().numpy())
        print("Balanced Accuracy:", balanced_acc)
        if wandb_enabled:
            import wandb
            wandb.log({"train/b_acc": balanced_acc})
        return False
    
    def save_callback(swipe_num, node):
        torch.save(layer.state_dict(), model_path)

    try:
        layer.tensor_network.accumulating_swipe(
            xinp_train, y_train, bf,
            batch_size=args.batch_size,
            num_swipes=args.num_swipes,
            method=args.method,
            lr=args.lr,
            eps=args.eps,
            orthonormalize=args.orthonormalize,
            convergence_criterion=convergence_criterion,
            timeout=None,  # let main process handle timeout
            verbose=args.verbose,
            block_callback=save_callback
        )
    finally:
        # Save checkpoint after each swipe
        torch.save(layer.state_dict(), model_path)

def main():
    parser = argparse.ArgumentParser(description='Convolutional Tensor Network Training')
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), default='mnist')
    parser.add_argument('--data_path', type=str, default=None, help='Override default data path')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--kernel', type=int, required=True, help='Kernel configuration index (see DATASETS for options)')
    parser.add_argument('--N', type=int, default=None, help='Number of carriages')
    parser.add_argument('--r', type=int, default=None, help='Bond dimension')
    parser.add_argument('--CB', type=int, default=None, help='Convolution bond')
    parser.add_argument('--num_swipes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='exact')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--orthonormalize', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    args = parser.parse_args()

    # Set CUDA device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset defaults
    ds = DATASETS[args.dataset]
    # Get kernel config from dictionary
    if args.kernel not in ds['kernels']:
        raise ValueError(f"Kernel index {args.kernel} not defined for dataset {args.dataset}. Available: {list(ds['kernels'].keys())}")
    kernel_size, stride, padding = ds['kernels'][args.kernel]
    N = args.N if args.N is not None else 3
    r = args.r if args.r is not None else 3
    CB = args.CB if args.CB is not None else -1

    # Data loading
    train_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, args.download, args.data_path)
    xinp_train, y_train = preprocess_batches(train_loader, num_classes, kernel_size, stride, device, padding=padding)

    # Use a temp file for model state
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        model_path = tmp.name

    # Start subprocess for training
    proc = multiprocessing.Process(target=train_with_timeout, args=(args, kernel_size, stride, padding, N, r, CB, xinp_train, y_train, model_path))
    proc.start()
    proc.join(args.timeout if args.timeout else None)
    if proc.is_alive():
        print(f"Timeout reached ({args.timeout} seconds). Terminating training process.")
        proc.terminate()
        proc.join()

    # Model setup for evaluation
    from tensor.layers import TensorConvolutionTrainLayer
    layer = TensorConvolutionTrainLayer(
        num_carriages=N,
        bond_dim=r,
        num_patches=xinp_train.shape[1],
        patch_pixels=xinp_train.shape[2],
        output_shape=y_train.shape[1]-1,
        convolution_bond=CB
    ).to(device)

    # Load trained weights if available
    try:
        state = torch.load(model_path, map_location=device)
        layer.load_state_dict(state)
        print("Loaded trained model state.")
    except Exception as e:
        print("Could not load trained model state:", e)

    # Load test data
    xinp_test, y_test = preprocess_batches(test_loader, num_classes, kernel_size, stride, device, padding=padding)

    # Test accuracy
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
    print('Test Acc:', accuracy_test)
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), reinit=True)
        wandb.log({"test/b_acc_f": accuracy_test})

if __name__ == '__main__':
    main()
