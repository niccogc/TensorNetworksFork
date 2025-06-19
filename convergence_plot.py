# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn.functional as F
import numpy as np
import time
torch.set_default_dtype(torch.float64)

dataset = 'higgs_small'
task = 'classification'
data_dir = '/work3/s183995/Tabular/data/processed'
data = torch.load(os.path.join(data_dir, dataset + '_tensor.pt'), weights_only=False)
X_train = data['X_train'].cuda()
X_test = data['X_test'].cuda()
X_val = data['X_val'].cuda()
y_train = data['y_train'].cuda()
y_test = data['y_test'].cuda()
y_val = data['y_val'].cuda()

num_classes = torch.unique(y_train).shape[0]

if task == 'classification':
    if y_train.ndim == 1:
        y_train = torch.nn.functional.one_hot(y_train.to(dtype=torch.long), num_classes=num_classes)
        y_val = torch.nn.functional.one_hot(y_val.to(dtype=torch.long), num_classes=num_classes)
        y_test = torch.nn.functional.one_hot(y_test.to(dtype=torch.long), num_classes=num_classes)

X_train = X_train.to(torch.float64)
y_train = y_train.to(torch.float64)
X_val = X_val.to(torch.float64)
y_val = y_val.to(torch.float64)
X_test = X_test.to(torch.float64)
y_test = y_test.to(torch.float64)

#%%
from tensor.layers import TensorTrainLayer
from tensor.bregman import SquareBregFunction, XEAutogradBregman

num_swipes = 10
N = 6
r = 6
p = X_train.shape[1]
C = y_train.shape[1]

epss = np.geomspace(5.0, 1e-2, 2*num_swipes).tolist()

if task == 'classification':
    C = num_classes - 1

# Define Bregman function
tt_layer = TensorTrainLayer(N, r, p, output_shape=(C,)).cuda()
node_states = tt_layer.node_states(detach=False)

if task == 'classification':
    with torch.inference_mode():
        y_pred = tt_layer(X_train[:64].cuda())
        w = 1/y_pred.std().item() if y_pred.std().item() > 0 else 1.0
    bf = XEAutogradBregman(w=w)
else:
    bf = SquareBregFunction()


tt_time = time.perf_counter()
tt_times = []
tt_losses = []
def loss_callback(num_swipes, node, loss):
    global tt_time
    val_start = time.perf_counter()
    # Calculate val loss, pause time while doing it, then resume timer
    with torch.inference_mode():
        y_pred = tt_layer(X_val)
        val_loss = bf.forward(y_pred, y_val, only_loss=True)
        val_loss = val_loss.mean().item()
    # Calculate time taken for validation
    val_time = time.perf_counter() - val_start
    tt_time = tt_time - val_time
    tt_times.append(time.perf_counter() - tt_time)
    tt_losses.append(val_loss)
    print('Loss:', tt_losses[-1], 'Time:', tt_times[-1])

# Train the model using Tensor Train
loss_callback(0, None, 0.0)
tt_layer.tensor_network.accumulating_swipe(X_train, y_train, bf, batch_size=2048, lr=1.0, loss_callback=loss_callback, orthonormalize=False, method='ridge_cholesky', eps=epss, verbose=True, num_swipes=num_swipes)
# Calculate val accuracy / rmse based on task
with torch.inference_mode():
    if task == 'classification':
        y_pred = tt_layer(X_val)
        y_pred = torch.cat((y_pred, torch.zeros_like(y_pred[:, :1])), dim=1)
        accuracy = (y_pred.argmax(dim=-1) == y_val.argmax(dim=-1)).float().mean().item()
        print('Validation Accuracy:', accuracy)
    else:
        y_pred = tt_layer(X_val)
        rmse = torch.sqrt(F.mse_loss(y_pred, y_val)).item()
        print('Validation RMSE:', rmse)
#%%
# Train all nodes using full SGD (AdamW)
from torch import optim
from torch import nn
sgd_layer = TensorTrainLayer(N, r, p, output_shape=(C,)).cuda()
sgd_layer.load_node_states(node_states, set_value=False)

batch_size = 512
# Map node_states to idx and then create parameter list
param_order = {i: idx for idx, i in enumerate(node_states.keys())}
params = nn.ParameterList([nn.Parameter(node_state.detach().clone()) for node_state in node_states.values()])
# Convert params into node_states again
param_states = {i: params[param_order[i]] for i in node_states.keys()}
optimizer = optim.AdamW(params.parameters(), lr=1e-3)

sgd_layer.load_node_states(param_states, set_value=True)

sgd_time = time.perf_counter()
sgd_times = []
sgd_losses = []

data_size = X_train.shape[0]
batches = (data_size + batch_size - 1) // batch_size # round up division
# Calculate val loss and time
val_start = time.perf_counter()
with torch.inference_mode():
    y_pred = sgd_layer(X_val)
    val_loss = bf.forward(y_pred, y_val, only_loss=True)
    val_loss = val_loss.mean().item()
val_time = time.perf_counter() - val_start
sgd_time = sgd_time - val_time
sgd_times.append(time.perf_counter() - sgd_time)
sgd_losses.append(val_loss)
print('Val Loss:', val_loss, 'Time:', sgd_times[-1])
for epoch in range(200):
    optimizer.zero_grad()
    total_loss = 0.0
    for b in range(batches):
        X_batch = X_train[b*batch_size:(b+1)*batch_size].cuda()
        y_batch = y_train[b*batch_size:(b+1)*batch_size].cuda()
        sgd_layer.tensor_network.reset_stacks()
        y_pred = sgd_layer(X_batch)
        loss = bf.forward(y_pred, y_batch, only_loss=True)
        loss = loss.mean()
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    # Calculate val loss and time
    val_start = time.perf_counter()
    with torch.inference_mode():
        y_pred = sgd_layer(X_val)
        val_loss = bf.forward(y_pred, y_val, only_loss=True)
        val_loss = val_loss.mean().item()
    val_time = time.perf_counter() - val_start
    sgd_time = sgd_time - val_time
    sgd_times.append(time.perf_counter() - sgd_time)
    sgd_losses.append(val_loss)
    print('Val Loss:', val_loss, 'Time:', sgd_times[-1])
    if sgd_times[-1] > tt_times[-1]:
        print('SGD is running longer')
        break
# %%
# Train blockwise using SGD
from torch import optim
from torch import nn
bsgd_layer = TensorTrainLayer(N, r, p, output_shape=(C,)).cuda()
bsgd_layer.load_node_states(node_states, set_value=False)

batch_size = 512

bsgd_time = time.perf_counter()
bsgd_times = []
bsgd_losses = []

num_swipes = 10

def loss_callback(num_swipes, node, loss):
    global bsgd_time
    val_start = time.perf_counter()
    # Calculate val loss, pause time while doing it, then resume timer
    with torch.inference_mode():
        y_pred = bsgd_layer(X_val)
        val_loss = bf.forward(y_pred, y_val, only_loss=True)
        val_loss = val_loss.mean().item()
    # Calculate time taken for validation
    val_time = time.perf_counter() - val_start
    bsgd_time = bsgd_time - val_time
    bsgd_times.append(time.perf_counter() - bsgd_time)
    bsgd_losses.append(val_loss)
    print('Loss:', bsgd_losses[-1], 'Time:', bsgd_times[-1])

data_size = X_train.shape[0]
batches = (data_size + batch_size - 1) // batch_size # round up division

num_node_repeats = 5
tn = bsgd_layer.tensor_network
node_l2r = None
node_r2l = None
loss_callback(0, None, 0.0)
for NS in range(num_swipes):
    for node_l2r in tn.train_nodes:
        if node_l2r in tn.node_indices and node_r2l in tn.node_indices and tn.node_indices[node_l2r] == tn.node_indices[node_r2l]:
            continue

        param = nn.Parameter(node_l2r.tensor)
        node_l2r.set_tensor(param)
        
        optimizer = optim.AdamW([param], lr=1e-3)

        for nr in range(num_node_repeats):
            optimizer.zero_grad()
            total_loss = 0.0
            for b in range(batches):
                X_batch = X_train[b*batch_size:(b+1)*batch_size].cuda()
                y_batch = y_train[b*batch_size:(b+1)*batch_size].cuda()
                tn.reset_stacks()
                y_pred = bsgd_layer(X_batch)
                loss = bf.forward(y_pred, y_batch, only_loss=True)
                loss = loss.mean()
                loss.backward()
                total_loss += loss.item()
            optimizer.step()
            loss_callback(NS, node_l2r, total_loss / batches)
        node_l2r.set_tensor(param.detach().clone())

        
    # RIGHT TO LEFT
    for node_r2l in tn.train_nodes:
        if node_l2r in tn.node_indices and node_r2l in tn.node_indices and tn.node_indices[node_l2r] == tn.node_indices[node_r2l]:
            continue

        param = nn.Parameter(node_r2l.tensor)
        node_r2l.set_tensor(param)
        optimizer = optim.AdamW([param], lr=1e-3)

        for nr in range(num_node_repeats):
            optimizer.zero_grad()
            total_loss = 0.0
            for b in range(batches):
                X_batch = X_train[b*batch_size:(b+1)*batch_size].cuda()
                y_batch = y_train[b*batch_size:(b+1)*batch_size].cuda()
                tn.reset_stacks()
                y_pred = bsgd_layer(X_batch)
                loss = bf.forward(y_pred, y_batch, only_loss=True)
                loss = loss.mean()
                loss.backward()
                total_loss += loss.mean().item()
            optimizer.step()
            loss_callback(NS, node_r2l, total_loss / batches)
        node_r2l.set_tensor(param.detach().clone())
    if bsgd_times[-1] > tt_times[-1]:
        print('Block SGD is running longer')
        break
#%%
# Plot the convergence
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(tt_times, tt_losses, label='Tensor Train', color='blue')
ax.plot(sgd_times, sgd_losses, label='SGD', color='orange')
ax.plot(bsgd_times, bsgd_losses, label='Block SGD', color='green')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Validation Loss')
ax.set_title('Convergence of Tensor Train, SGD and Block SGD')
ax.legend()
plt.show()
# %%