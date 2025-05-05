import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_path = "/work3/s183995/CIFAR10/data"

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True)

    train_samples = []
    train_labels = []
    for images, labels in train_loader:
        train_samples.append(images)
        train_labels.append(labels)

    xinp_train = torch.cat(train_samples, dim=0).cuda()
    y_train = torch.cat(train_labels, dim=0).cuda()

    KERNEL_SIZE = 4
    STRIDE = 4

    xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
    xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2)
    xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1), device=xinp_train.device)), dim=-1)
    xinp_train[..., -1, -1] = 1.0
    y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64)


    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_samples = []
    test_labels = []
    for images, labels in test_loader:
        test_samples.append(images)
        test_labels.append(labels)
    xinp_test = torch.cat(test_samples, dim=0).cuda()
    y_test = torch.cat(test_labels, dim=0).cuda()
    xinp_test = F.unfold(xinp_test, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
    xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]), device=xinp_test.device)), dim=-2)
    xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1), device=xinp_test.device)), dim=-1)
    xinp_test[..., -1, -1] = 1.0
    y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64)


    num_swipes = 3

    N = 3
    r = 50
    CB = -1
    batch_size = 256
    max_iter = 2000

    # Define Bregman function
    layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1]-1, convolution_bond=CB).cuda()
    print('Num params:', layer.num_parameters())
    from tensor.utils import visualize_tensornetwork
    visualize_tensornetwork(layer.tensor_network)
    with torch.inference_mode():
        y_pred = layer(xinp_train[:64])
        w = 1/y_pred.std().item()
        #del y_pred
    bf = XEAutogradBregman(w=w)

    wandb.init(project="CIFARTT", config={
        "num_carriages": N,
        "bond_dim": r,
        "num_patches": xinp_train.shape[1],
        "patch_pixels": xinp_train.shape[2],
        "output_shape": y_train.shape[1]-1,
        "convolution_bond": CB,
        "batch_size": batch_size,
        "num_swipes": num_swipes,
        "lr": 1.0,
        "max_iter": max_iter,
        "w": w,
        "dataset": "CIFAR10",
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE,
        "num_params": layer.num_parameters(),
    })

    def convergence_criterion(NS, node):
        y_pred_test = layer(xinp_test)
        y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
        accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
        print(NS, node.name, 'Test Acc:', accuracy_test)
        wandb.log({"test_accuracy": accuracy_test,"swipe": NS})
        return False
    
    def loss_callback(loss):
        wandb.log({"loss": loss})
        return False

    layer.tensor_network.lanczos_swipe(xinp_train, y_train, bf, batch_size=batch_size, num_swipes=num_swipes, lr=1.0, max_iter=max_iter, verbose=2, block_callback=convergence_criterion, loss_callback=loss_callback)