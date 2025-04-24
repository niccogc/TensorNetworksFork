import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# ---- Dataset registry ----
DATASETS = {
    'mnist': {
        'class': torchvision.datasets.MNIST,
        'root': '/work3/s183995/MNIST/data',
        'num_classes': 10,
        'input_channels': 1,
        'normalize': ((0.1307,), (0.3081,)),
        'default_kernel': 2,
        'default_stride': 2,
    },
    'cifar10': {
        'class': torchvision.datasets.CIFAR10,
        'root': '/work3/s183995/CIFAR10/data',
        'num_classes': 10,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'default_kernel': 8,
        'default_stride': 8,
    },
    'cifar100': {
        'class': torchvision.datasets.CIFAR100,
        'root': '/work3/s183995/CIFAR100/data',
        'num_classes': 100,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'default_kernel': 8,
        'default_stride': 8,
    },
    'imagenet': {
        'class': torchvision.datasets.ImageNet,
        'root': '/work3/s183995/ImageNet/data',
        'num_classes': 100,
        'input_channels': 3,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'default_kernel': 8,
        'default_stride': 8,
    },
}

def get_data_loaders(dataset_name, batch_size, kernel_size, stride, download, data_path=None):
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

def main():
    parser = argparse.ArgumentParser(description='Convolutional Tensor Network Training')
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), default='mnist')
    parser.add_argument('--data_path', type=str, default=None, help='Override default data path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--padding', type=int, default=0, help='Padding for convolution')
    parser.add_argument('--N', type=int, default=None, help='Number of carriages')
    parser.add_argument('--r', type=int, default=None, help='Bond dimension')
    parser.add_argument('--CB', type=int, default=None, help='Convolution bond')
    parser.add_argument('--num_swipes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--method', type=str, default='exact')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--orthonormalize', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for training')
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset defaults
    ds = DATASETS[args.dataset]
    kernel_size = args.kernel_size if args.kernel_size is not None else ds['default_kernel']
    stride = args.stride if args.stride is not None else ds['default_stride']
    padding = args.padding
    N = args.N if args.N is not None else (2 if args.dataset == 'mnist' else 3)
    r = args.r if args.r is not None else (2 if args.dataset == 'mnist' else 8)
    CB = args.CB if args.CB is not None else (-1 if args.dataset == 'mnist' else 8)

    # Data loading
    train_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, kernel_size, stride, args.download, args.data_path)
    xinp_train, y_train = preprocess_batches(train_loader, num_classes, kernel_size, stride, device, padding=padding)

    # Model setup
    layer = TensorConvolutionTrainLayer(
        num_carriages=N,
        bond_dim=r,
        num_patches=xinp_train.shape[1],
        patch_pixels=xinp_train.shape[2],
        output_shape=y_train.shape[1]-1,
        convolution_bond=CB
    ).to(device)
    print('Num params:', layer.num_parameters())

    # Bregman function
    with torch.inference_mode():
        y_pred = layer(xinp_train[:64])
        w = 1/y_pred.std().item()
    bf = XEAutogradBregman(w=w)

    def convergence_criterion(y_pred, y_true):
        y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
        balanced_acc = balanced_accuracy_score(y_true.argmax(dim=-1).cpu().numpy(), y_pred.argmax(dim=-1).cpu().numpy())
        print("Balanced Accuracy:", balanced_acc)
        return False

    # Training
    layer.tensor_network.accumulating_swipe(
        xinp_train, y_train, bf,
        batch_size=args.batch_size,
        lr=args.lr,
        convergence_criterion=convergence_criterion,
        orthonormalize=args.orthonormalize,
        method=args.method,
        eps=args.eps,
        verbose=args.verbose,
        num_swipes=args.num_swipes,
        timeout=args.timeout
    )

    # Train accuracy
    y_pred_train = layer(xinp_train)
    y_pred_train = torch.cat((y_pred_train, torch.zeros_like(y_pred_train[:, :1])), dim=1)
    accuracy_train = balanced_accuracy_score(y_train.argmax(dim=-1).cpu().numpy(), y_pred_train.argmax(dim=-1).cpu().numpy())
    print('Train Acc:', accuracy_train)

    # Load test data
    xinp_test, y_test = preprocess_batches(test_loader, num_classes, kernel_size, stride, device, padding=padding)

    # Test accuracy
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
    print('Test Acc:', accuracy_test)

if __name__ == '__main__':
    main()
