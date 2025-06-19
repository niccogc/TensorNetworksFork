#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
import nonlinear_benchmarks
train_val, test = nonlinear_benchmarks.Cascaded_Tanks(dir_placement='/work3/s183995/Nonlinear/')
X_train, y_train = train_val
X_test, y_test = test
X_train = np.vstack([np.linspace(0, 1, X_train.shape[0]),]).T
X_test = np.vstack([np.linspace(0, 1, X_test.shape[0]),]).T
#%%
import torch
from sklearn.model_selection import train_test_split
# data (as pandas dataframes) 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert to torch tensors (float64 for compatibility)
X_train = torch.tensor(X_train, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
X_val = torch.tensor(X_val, dtype=torch.float64)
y_val = torch.tensor(y_val, dtype=torch.float64)
X_test = torch.tensor(X_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

# Save in the required format
out = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}
torch.save(out, '/work3/s183995/Nonlinear/data/cascaded_tanks_tensor.pt')
#%%
