#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
torch.set_default_dtype(torch.bfloat16)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="/work3/s183995/MNIST/data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

train_samples = []
train_labels = []
for images, labels in train_loader:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0)
y_train = torch.cat(train_labels, dim=0)

KERNEL_SIZE = 4
STRIDE = 4

xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2).cuda()
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1), device=xinp_train.device)), dim=-1).cuda()
xinp_train[..., -1, -1] = 1.0
y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64).cuda()


test_dataset = torchvision.datasets.MNIST(root="/work3/s183995/MNIST/data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
test_samples = []
test_labels = []
for images, labels in test_loader:
    test_samples.append(images)
    test_labels.append(labels)
xinp_test = torch.cat(test_samples, dim=0)
y_test = torch.cat(test_labels, dim=0)
xinp_test = F.unfold(xinp_test, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]), device=xinp_test.device)), dim=-2).cuda()
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1), device=xinp_test.device)), dim=-1).cuda()
xinp_test[..., -1, -1] = 1.0
y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64).cuda()

# %%
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from matplotlib import pyplot as plt

num_swipes = 10
epss = np.geomspace(1.0, 1e-2, 2*num_swipes).tolist()
plt.plot(epss)

N = 3
r = 16
CB = 4
batch_size = 512
max_iter = 500
tol = 1e-3

def convergence_criterion(NS, node):
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
    print(NS, node.name,'Test Acc:', accuracy_test)
    return False

# Define Bregman function
layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1]-1, convolution_bond=CB).cuda()
print('Num params:', layer.num_parameters())
#%%
from tensor.utils import visualize_tensornetwork
visualize_tensornetwork(layer.tensor_network)
#%%
with torch.inference_mode():
    y_pred = layer(xinp_train[:64])
    w = 1/y_pred.std().item()
    del y_pred
bf = XEAutogradBregman(w=w)
#%%
from scipy.sparse.linalg import minres

def loss_callback(loss):
    print("loss", loss)
    return False

layer.tensor_network.scipy_swipe(xinp_train, y_train, bf, minres, batch_size=1024, num_swipes=num_swipes, lr=1.0, max_iter=max_iter, tol=tol, verbose=2, loss_callback=loss_callback, block_callback=convergence_criterion)
#%%
