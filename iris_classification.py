# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)

iris = datasets.load_iris()
X = iris['data']  # shape (150, 4)
y = iris['target']  # shape (150,)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch
X = torch.tensor(X_scaled, dtype=torch.float64)
y = torch.tensor(y, dtype=torch.long)

# One-hot encode labels
y = F.one_hot(y, num_classes=3).to(dtype=torch.float64)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Move data to GPU
X_train, X_test = X_train.cuda(), X_test.cuda()
y_train, y_test = y_train.cuda(), y_test.cuda()

# Prepare train input
xinp_train = torch.cat([torch.ones(X_train.shape[0], 1, dtype=X_train.dtype, device=X_train.device), X_train], dim=-1)
xinp_test = torch.cat([torch.ones(X_test.shape[0], 1, dtype=X_test.dtype, device=X_test.device), X_test], dim=-1)

#%%
from tensor.layers import TensorTrainLayer
from tensor.bregman import KLDivBregman, AutogradBregman

N = 2
r = 2
p = X.shape[1]+1
C = y.shape[1]

# Define Bregman function
layer = TensorTrainLayer(N, r, p, output_shape=C).cuda()
y_pred = layer(xinp_train)
w = 0.1*1/y_pred.std().item()
bf = KLDivBregman(w=w)

#PyTorch Autograd Bregman (very unstable for now)
forward_transform = lambda x, y: (torch.softmax(w*x, dim=-1), y) # y is already softmax (one-hot)
phi_func = lambda x: torch.sum(torch.where(x == 0, torch.zeros_like(x), x * torch.log(x)), dim=-1, keepdim=True) #0 * log(0) = 0 assumption
d_phi_x_func = lambda x: torch.log(x) + 1
bf_auto = AutogradBregman(phi_func=phi_func, forward_transform=forward_transform, d_phi_x_func=d_phi_x_func)

def convergence_criterion(y_pred, y_true):
    accuracy = (y_pred.argmax(dim=-1) == y_true.argmax(dim=-1)).float().mean().item()
    print("Accuracy:", accuracy)
    return accuracy > 0.95
#%%
# Test bf vs. bf_auto
y_pred = layer(xinp_train)
loss_auto, d_loss_auto, dd_loss_auto = bf_auto(y_pred, y_train)
loss, d_loss, dd_loss = bf(y_pred, y_train)
print('Loss:', loss.mean().item())
print('Loss auto:', loss_auto.mean().item())
print("Loss allclose:", torch.allclose(loss, loss_auto))
print("d Loss allclose:", torch.allclose(d_loss, d_loss_auto))
print("dd Loss allclose:", torch.allclose(dd_loss, dd_loss_auto))
#%%
# Train the model
layer.tensor_network.swipe(xinp_train, y_train, bf, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=True, method='exact', eps=1e-5, verbose=True, num_swipes=100)
#%%
# Calculate accuracy on train set
y_pred_train = layer(xinp_train)
accuracy_train = (y_pred_train.argmax(dim=-1) == y_train.argmax(dim=-1)).float().mean().item()
print('Train Acc:', accuracy_train)

# Calculate accuracy on test set
y_pred_test = layer(xinp_test)
accuracy_test = (y_pred_test.argmax(dim=-1) == y_test.argmax(dim=-1)).float().mean().item()
print('Test Acc:', accuracy_test)
#%%