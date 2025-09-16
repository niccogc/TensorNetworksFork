#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torchvision.transforms as transforms
torch.set_default_dtype(torch.float64)
from tensor.layers import TensorConvolutionTrainLayer, TensorNetworkLayer
from tensor.network import SumOfNetworks
from tensor.bregman import XEAutogradBregman
from models.tensor_train import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
from torch.nn import functional as F

def error_rate_torch(y_true, y_pred):
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    if y_true_labels.ndim > 1 and y_true_labels.shape[1] > 1:
        y_true_labels = np.argmax(y_true_labels, axis=1)
    return 1.0 - accuracy_score(y_true_labels, y_pred_labels)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = 'MNIST'
num_swipes = 5
N = 4
r = 10
CB = 4
eps = 5.0
eps_decay = 0.25
model_type = 'tt_type1'
kernel_size = 4
kernel_stride = 4
abs_err = 1e-4
rel_err = 1e-3
early_stopping = 10
seed = 42
validation_split = 0.1
verbose = 1

if dataset == 'MNIST':
    train_dataset = torchvision.datasets.MNIST(root="/work3/aveno/MNIST/data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root="/work3/aveno/MNIST/data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
elif dataset == 'FashionMNIST':
    train_dataset = torchvision.datasets.FashionMNIST(root="/work3/aveno/FashionMNIST/data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.FashionMNIST(root="/work3/aveno/FashionMNIST/data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

train_samples = []
train_labels = []
for images, labels in train_loader:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0)
y_train = torch.cat(train_labels, dim=0)

xinp_train = F.unfold(xinp_train, kernel_size=(kernel_size,kernel_size), stride=(kernel_stride,kernel_stride), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]), device=xinp_train.device)), dim=-2).cuda()
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1), device=xinp_train.device)), dim=-1).cuda()
xinp_train[..., -1, -1] = 1.0
y_train = F.one_hot(y_train, num_classes=10).to(dtype=torch.float64).cuda()

# Do a random split of train into train and val
n = xinp_train.shape[0]
idx = np.arange(n)
rng = np.random.RandomState(seed)
rng.shuffle(idx)
split = int(n * (1 - validation_split))
train_idx, val_idx = idx[:split], idx[split:]
xinp_train, xinp_val = xinp_train[train_idx], xinp_train[val_idx]
y_train, y_val = y_train[train_idx], y_train[val_idx]

test_samples = []
test_labels = []
for images, labels in test_loader:
    test_samples.append(images)
    test_labels.append(labels)
xinp_test = torch.cat(test_samples, dim=0)
y_test = torch.cat(test_labels, dim=0)
xinp_test = F.unfold(xinp_test, kernel_size=(kernel_size,kernel_size), stride=(kernel_stride,kernel_stride), padding=0).transpose(-2, -1)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]), device=xinp_test.device)), dim=-2).cuda()
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1), device=xinp_test.device)), dim=-1).cuda()
xinp_test[..., -1, -1] = 1.0
y_test = F.one_hot(y_test, num_classes=10).to(dtype=torch.float64).cuda()

if 'type1' in model_type:
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
            train_operators=True
        )
    ).cuda()
else:
    layer = TensorConvolutionTrainLayer(num_carriages=N, bond_dim=r, num_patches=xinp_train.shape[1], patch_pixels=xinp_train.shape[2], output_shape=y_train.shape[1]-1, convolution_bond=CB)
print('Num params:', layer.num_parameters())

# Define early stopping

def model_predict(x):
    y_pred = layer(x)
    y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
    return y_pred

early_stopper = EarlyStopping(
    xinp_val, y_val,
    model_predict=model_predict,
    get_model_weights=layer.node_states,
    loss_fn=error_rate_torch,
    abs_err=abs_err,
    rel_err=rel_err,
    early_stopping=early_stopping,
    verbose=verbose
)
# Define Bregman function
with torch.inference_mode():
    y_pred = layer.forward(xinp_train[:64], to_tensor=True)
    w = 1/y_pred.std().item()
    #del y_pred
bf = XEAutogradBregman(w=w)
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, batch_size=2048, lr=1.0, convergence_criterion=early_stopper.convergence_criterion, orthonormalize=False, method='ridge_cholesky', eps=eps, eps_decay=eps_decay, verbose=verbose, num_swipes=num_swipes, disable_tqdm=verbose < 3)

# Reload best model
layer.load_node_states(early_stopper.best_state_dict, set_value=True)

# Evaluate on test set
with torch.inference_mode():
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
print(f"{dataset},{model_type},{N},{r},{CB},{accuracy_test*100:.2f},{num_swipes},{eps},{eps_decay},{kernel_size},{kernel_stride},{early_stopping},{abs_err},{rel_err},{seed},{validation_split}")
#%%