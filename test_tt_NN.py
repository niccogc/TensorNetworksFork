#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import sklearn.preprocessing as skpp
import sklearn.datasets as skds
import sklearn.model_selection as skms
from sklearn.preprocessing import QuantileTransformer

def load_openml(name, y_dict = None):
    df = skds.fetch_openml(name=name,as_frame=True)
    X = df.data.to_numpy(dtype=np.float64)     
    y = df.target
    if y_dict is not None: y=y.astype("str").map(y_dict)
    y = y.to_numpy(dtype=np.float64)
    return X, y

def load_data():
    X, y = load_openml("house_16H")
    
    if len(y.shape)==1: 
        y = y[:,np.newaxis]
    
    y_scaler = skpp.StandardScaler()
    y = y_scaler.fit_transform(y)

    return X, y

X, y = load_data()
X_train, X_val, y_train, y_val = skms.train_test_split(X, y, test_size=0.2, random_state=42)

X_quant = QuantileTransformer(output_distribution="uniform",subsample=1_000_000,random_state=0)
X_train = X_quant.fit_transform(X_train)
X_val = X_quant.transform(X_val)
#%%
import torch
from tensor.layers import TensorTrainNN, tensor_network_update
from sklearn.metrics import root_mean_squared_error, r2_score

MLP = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 64),
    #torch.nn.ReLU(),
    #torch.nn.Linear(64, 32),
    #torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    TensorTrainNN(64, 32, N=10, r=16, perturb=True, natural_gradient=False),
    torch.nn.Linear(32, y_train.shape[1]),
).cuda()

batch_size = 512
opt = torch.optim.Adam(MLP.parameters(), lr=1e-3)
MLP.zero_grad(set_to_none=True)
for epoch in range(100):
    # Random perm
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), batch_size):
        x_batch = torch.tensor(X_train[perm[i:i+batch_size]], dtype=torch.float32).cuda()
        y_batch = torch.tensor(y_train[perm[i:i+batch_size]], dtype=torch.float32).cuda()

        y_pred = MLP(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        loss.backward(create_graph=True, retain_graph=True)
        opt.step()
        MLP.zero_grad(set_to_none=True)
    #MLP.apply(tensor_network_update)
    # Validation
    with torch.no_grad():
        x_val_tensor = torch.tensor(X_val, dtype=torch.float32).cuda()
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).cuda()
        y_val_pred = MLP(x_val_tensor)
        val_loss = root_mean_squared_error(y_val_tensor.cpu().numpy(), y_val_pred.cpu().numpy())
        r2 = r2_score(y_val_tensor.cpu().numpy(), y_val_pred.cpu().numpy())
        print(f"Epoch {epoch+1}, Validation: RMSE: {val_loss:.4f}, R2: {r2:.4f}")
#%%