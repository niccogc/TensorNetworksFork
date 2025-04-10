#%%
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
    break
xinp_train = torch.cat(train_samples, dim=0)
y_train = torch.cat(train_labels, dim=0)

xinp_train = F.unfold(xinp_train, kernel_size=(7,7), stride=(7,7), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.ones((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2)
y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64)

# %%
from tensor.layers import TensorConvolutionTrainLayer
from tensor.bregman import KLDivBregman
N = 2
r = 3

# Define Bregman function
layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1])
y_pred = layer(xinp_train)
w = 0.1*1/y_pred.std().item()
bf = KLDivBregman(w=w)

def convergence_criterion(y_pred, y_true):
    accuracy = (y_pred.argmax(dim=-1) == y_true.argmax(dim=-1)).float().mean().item()
    print("Accuracy:", accuracy)
    return accuracy > 0.95
#%%
# Train the model
layer.tensor_network.swipe(xinp_train, y_train, bf, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=True, method='exact', eps=1e-5, verbose=True, num_swipes=100)
#%%
# Calculate accuracy on train set
y_pred_train = layer(xinp_train)
accuracy_train = (y_pred_train.argmax(dim=-1) == y_train.argmax(dim=-1)).float().mean().item()
print('Train Acc:', accuracy_train)
#%%