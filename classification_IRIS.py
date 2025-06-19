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
from tensor.layers import TensorTrainLinearLayer, TensorTrainLinearLayer
from tensor.bregman import XEAutogradBregman

N = 10
r = 100
p = X.shape[1]+1
C = y.shape[1]-1

# Define Bregman function
layer = TensorTrainLinearLayer(N, r, p, output_shape=(C,), linear_bond=2, linear_dim=5).cuda()
#layer = TensorTrainLinearLayer(N, r, p, 2, 2, output_shape=(C,), connect_linear=False).cuda()
#%%
y_pred = layer(xinp_train)
w = 1/y_pred.std().item()
bf = XEAutogradBregman(w=w)
#%%
from tensor.utils import visualize_tensornetwork
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(20, 20))
visualize_tensornetwork(layer.tensor_network, fig=fig, ax=ax)
#%%
from sklearn.metrics import balanced_accuracy_score
def convergence_criterion(*args):
    y_pred_test = layer(xinp_test)
    y_pred_test = torch.cat((y_pred_test, torch.zeros_like(y_pred_test[:, :1])), dim=1)
    accuracy_test = balanced_accuracy_score(y_test.argmax(dim=-1).cpu().numpy(), y_pred_test.argmax(dim=-1).cpu().numpy())
    print('Test Acc:', accuracy_test)
    return False

# Train the model
layer.tensor_network.accumulating_swipe(xinp_train, y_train, bf, lr=1.0, convergence_criterion=convergence_criterion, orthonormalize=False, method='ridge_exact', eps=0.2, verbose=True, num_swipes=10)
#%%
# Calculate accuracy on train set
y_pred_train = layer(xinp_train)
print("Train")
convergence_criterion(y_pred_train, y_train)
y_pred_test = layer(xinp_test)
print("Test")
convergence_criterion(y_pred_test, y_test)
None
#%%