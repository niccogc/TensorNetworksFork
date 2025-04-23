#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

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

KERNEL_SIZE = 7
STRIDE = 7

xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.ones((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2).cuda()
y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64).cuda()

# %%
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score

N = 3
r = 14
CB = 8

def convergence_criterion(y_pred, y_true):
    y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
    balanced_acc = balanced_accuracy_score(y_true.argmax(dim=-1).cpu().numpy(), y_pred.argmax(dim=-1).cpu().numpy())
    print("Balanced Accuracy:", balanced_acc)
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
    #del y_pred
bf = XEAutogradBregman(w=w)
#%%
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, batch_size=512, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method='exact', eps=1e-4, verbose=2, num_swipes=100)
#%%
# Calculate accuracy on train set
y_pred_train = layer(xinp_train)
y_pred_train = torch.cat((y_pred_train, torch.zeros_like(y_pred_train[:, :1])), dim=1)
accuracy_train = balanced_accuracy_score(y_train.argmax(dim=-1).cpu().numpy(), y_pred_train.argmax(dim=-1).cpu().numpy())
print('Train Acc:', accuracy_train)
#%%
from matplotlib import pyplot as plt
plt.imshow(layer.conv_blocks[0].tensor[:,6].view(KERNEL_SIZE,KERNEL_SIZE).cpu().numpy())
#%%
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
xinp_test = torch.cat((xinp_test, torch.ones((xinp_test.shape[0], 1, xinp_test.shape[2]), device=xinp_test.device)), dim=-2).cuda()
y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64).cuda()
#%%
# Calculate accuracy on test set
y_pred_test = layer(xinp_test)
y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
print('Test Acc:', accuracy_test)
#%%
img_idx = 5
block_idx = 0
conv = (xinp_train[img_idx] @ layer.conv_blocks[block_idx].tensor)[:16].view(4,4,-1)
fig, axs = plt.subplots(4,2)
for i, ax in enumerate(axs.flatten()):
    ax.imshow(conv[:,:,i].cpu().numpy())
    ax.axis('off')
plt.tight_layout()
plt.show()
from torch.nn import Fold
plt.imshow(Fold(output_size=(28,28), kernel_size=(7,7), stride=(7,7))(xinp_train[img_idx][:16].T)[0].cpu().numpy())
# %%
