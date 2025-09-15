#%%
import os
from scipy.io import savemat
from load_ucirepo import get_ucidata, datasets

path = '/work3/aveno/Tabular/mat/'
if not os.path.exists(path):
    os.makedirs(path)

for dataset, dataset_id, task in datasets:
    X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(dataset_id, task, device='cpu')
    mat_dict = {'X_train': X_train.numpy(), 'y_train': y_train.numpy(),
                'X_val': X_val.numpy(), 'y_val': y_val.numpy(),
                'X_test': X_test.numpy(), 'y_test': y_test.numpy()}
    savemat(os.path.join(path, f'{dataset}.mat'), mat_dict)
    print(f"Converted {dataset} to .mat format.")
# %%
    