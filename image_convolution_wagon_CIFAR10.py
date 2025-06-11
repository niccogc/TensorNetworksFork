#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = "/work3/s183995/CIFAR10/data"

train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

train_samples = []
train_labels = []
for images, labels in train_loader:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0).cuda()
y_train = torch.cat(train_labels, dim=0).cuda()

KERNEL_SIZE = 4
STRIDE = 4

def prep_data(xinp):
    N, C, H, W = xinp.shape
    KH = KW = KERNEL_SIZE
    PH = H // STRIDE
    PW = W // STRIDE

    # Unfold and reshape
    xinp = F.unfold(xinp, kernel_size=(KH, KW), stride=(STRIDE, STRIDE), padding=0)  # (N, C*KH*KW, L)
    xinp = xinp.reshape(N, C, KH, KW, PH, PW).permute(0, 4, 5, 2, 3, 1)

    # Pad each dimension after the batch
    for dim in range(1, xinp.dim()):
        pad_shape = list(xinp.shape)
        pad_shape[dim] = 1  # Insert 1 along current dim
        pad = torch.zeros(pad_shape, device=xinp.device, dtype=xinp.dtype)
        xinp = torch.cat((xinp, pad), dim=dim)

    # Set the final corner to 1
    xinp[..., *([-1]*(xinp.ndim-1))] = 1.0

    return xinp.contiguous()
xinp_train = prep_data(xinp_train)
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
xinp_test = prep_data(xinp_test)
y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64)
# %%
from tensor.layers import TensorTrainSplitInputLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from matplotlib import pyplot as plt

num_swipes = 20
epss = np.geomspace(1e-1, 1e-8, 2*num_swipes).tolist()
plt.plot(epss)

N = 3
r = 10
AB = 12

def convergence_criterion(*args):
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
    print('Test Acc:', accuracy_test)
    return False

# Define Bregman function
layer = TensorTrainSplitInputLayer(num_wagons=N, bond_dim=r, input_shape=xinp_train.shape[1:], output_shape=(y_train.shape[1]-1,), axle_bond=AB).cuda()
print('Num params:', layer.num_parameters())
#%%
from tensor.utils import visualize_tensornetwork
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
visualize_tensornetwork(layer.tensor_network, fig=fig, ax=ax)
#%%
with torch.inference_mode():
    y_pred = layer(xinp_train[:64])
    w = 1/y_pred.std().item()
bf = XEAutogradBregman(w=w)
#%%
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, batch_size=512, delta=3.0, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method='ridge_exact', eps=epss, verbose=2, num_swipes=num_swipes, disable_tqdm=True)
#%%