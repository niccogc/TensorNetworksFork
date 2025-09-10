#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
torch.set_default_dtype(torch.float64)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="/work3/aveno/MNIST/data", train=True, transform=transform, download=True)
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

# Do a random split of train into train and val
seed = 42
validation_split = 0.1
n = xinp_train.shape[0]
idx = np.arange(n)
rng = np.random.RandomState(seed)
rng.shuffle(idx)
split = int(n * (1 - validation_split))
train_idx, val_idx = idx[:split], idx[split:]
xinp_train, xinp_val = xinp_train[train_idx], xinp_train[val_idx]
y_train, y_val = y_train[train_idx], y_train[val_idx]

test_dataset = torchvision.datasets.MNIST(root="/work3/aveno/MNIST/data", train=False, transform=transform, download=True)
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
from tensor.layers import TensorConvolutionTrainLayer, TensorNetworkLayer
from tensor.network import SumOfNetworks
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt

num_swipes = 3

N = 3
r = 14
CB = 6
trajectory = []
epoch = 0
def convergence_criterion():
    global epoch
    epoch += 1
    y_pred_val = layer(xinp_val)
    y_pred_val = torch.cat((y_pred_val, torch.zeros_like(y_pred_val[:, :1])), dim=1)
    accuracy_test = accuracy_score(y_val.argmax(dim=-1).cpu().numpy(), y_pred_val.argmax(dim=-1).cpu().numpy())
    trajectory.append({
        'epoch': epoch,
        'val_accuracy': accuracy_test
    })
    print(f"Epoch {epoch}, val accuracy: {accuracy_test}")
    return False

# Define Bregman function
nets = []
for i in range(1, N+1):
    if i == 1:
        num_patches = xinp_train.shape[1]
        patch_pixels = xinp_train.shape[2]
    else:
        num_patches = xinp_train.shape[1] - 1
        patch_pixels = xinp_train.shape[2] - 1
    net = TensorConvolutionTrainLayer(num_carriages=i, bond_dim=r, num_patches=num_patches, patch_pixels=patch_pixels, output_shape=y_train.shape[1]-1, convolution_bond=CB).tensor_network
    nets.append(net)
layer = TensorNetworkLayer(
    SumOfNetworks(
        nets,
        only_bias_first=True,
        train_linear=True
    )
).cuda()
print('Num params:', layer.num_parameters())
#%%
from tensor.utils import visualize_tensornetwork
visualize_tensornetwork(layer.tensor_network)
#%%
with torch.inference_mode():
    y_pred = layer.forward(xinp_train[:64], to_tensor=True)
    w = 1/y_pred.std().item()
    #del y_pred
bf = XEAutogradBregman(w=w)
#%%
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, eps=1.0, eps_decay=0.5, batch_size=2048, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method='ridge_exact', verbose=2, num_swipes=num_swipes)
#%%
import numpy as np
import pandas as pd
data = []
for traj in trajectory:
    data.append((traj['epoch'], traj['val_accuracy']))

df = pd.DataFrame(data, columns=['Epoch', 'Val Accuracy'])
df.to_csv(f'tt_convolution_N{N}_r{r}_cb{CB}_swipes{num_swipes}_P{layer.num_parameters()}_fit_mnist.csv', index=False)
# %%
