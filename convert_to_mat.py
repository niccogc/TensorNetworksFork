#%%
import torch
from glob import glob
from scipy.io import savemat

def load_tabular_data(filename, device):
    data = torch.load(filename, map_location=device)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_val = data['X_val'].to(device)
    y_val = data['y_val'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    if 'processed' in filename:
        print("Processing data for tabular model...")
        X_train = torch.cat((X_train, torch.ones((X_train.shape[0], 1), device=X_train.device)), dim=-1)
        X_val = torch.cat((X_val, torch.ones((X_val.shape[0], 1), device=X_val.device)), dim=-1)
        X_test = torch.cat((X_test, torch.ones((X_test.shape[0], 1), device=X_test.device)), dim=-1)
    return X_train, y_train, X_val, y_val, X_test, y_test

path = '/work3/aveno/Tabular/data/processed/'

files = glob(path + '*_tensor.pt')
for file in files:
    X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_data(file, device='cpu')
    # Check if the last feature is all ones, and if so, remove it
    if torch.all(X_train[:, -1] == 1) and torch.all(X_val[:, -1] == 1) and torch.all(X_test[:, -1] == 1):
        print("Removing last feature (bias term) from data...")
        X_train = X_train[:, :-1]
        X_val = X_val[:, :-1]
        X_test = X_test[:, :-1]
    mat_dict = {'X_train': X_train.numpy(), 'y_train': y_train.numpy(),
                'X_val': X_val.numpy(), 'y_val': y_val.numpy(),
                'X_test': X_test.numpy(), 'y_test': y_test.numpy()}
    savemat(file.replace('.pt', '.mat'), mat_dict)
    print(f"Converted {file} to .mat format.")
# %%
    