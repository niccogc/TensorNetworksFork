#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = "/work3/s183995/CIFAR100/data"

train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                    download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

train_samples = []
train_labels = []
for images, labels in train_dataloader:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0)
y_train = torch.cat(train_labels, dim=0)

KERNEL_SIZE = 8
STRIDE = 8

xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.ones((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2).cuda()
y_train = F.one_hot(y_train, num_classes=100).to(dtype=torch.float64).cuda()

# %%
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import XEAutogradBregman
from sklearn.metrics import balanced_accuracy_score, accuracy_score

N = 3
r = 4
CB = 10

def convergence_criterion(y_pred, y_true):
    y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
    balanced_acc = balanced_accuracy_score(y_true.argmax(dim=-1).cpu().numpy(), y_pred.argmax(dim=-1).cpu().numpy())
    print("Balanced Accuracy:", balanced_acc)
    return balanced_acc > 0.95

# Define Bregman function
layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1]-1, convolution_bond=CB).cuda()
print("Num params:", layer.num_parameters())
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
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, batch_size=512, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method='exact', eps=1e-4, verbose=2, num_swipes=1)
#%%
# Calculate accuracy on train set
y_pred_train = layer(xinp_train)
y_pred_train = torch.cat((y_pred_train, torch.zeros_like(y_pred_train[:, :1])), dim=1)
accuracy_train = balanced_accuracy_score(y_train.argmax(dim=-1).cpu().numpy(), y_pred_train.argmax(dim=-1).cpu().numpy())
print('Train Acc:', accuracy_train)
#%%
test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                        download=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                            shuffle=False)

test_samples = []
test_labels = []
for images, labels in test_dataloader:
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
print('Balanced Test Acc:', accuracy_test)
# Also just calculate standard accuracy
accuracy_test_standard = accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
print('Standard Test Acc:', accuracy_test_standard)
#%%