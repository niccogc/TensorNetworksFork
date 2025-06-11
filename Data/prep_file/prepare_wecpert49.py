from ucimlrepo import fetch_ucirepo 
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

X = pd.read_csv('../raw/WEC/WEC_Perth_49.csv')
# y = X.pop()
individual_power = [f'Power{i}' for i in range(1,50)]
tot_power = ['Total_Power']
y_individual = X[individual_power]
X = X.drop(columns=individual_power)
y_total = X.pop(*tot_power)
y = y_total
print(X.head())
print(y.head())
# Standardize features
scaler = StandardScaler()
y_onehot = scaler.fit_transform(y.values.reshape(-1,1))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
# print(X)
# print(y)
# Split into train, val, test (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_onehot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(y_test)
# Convert to torch tensors (float64 for compatibility)
X_train = torch.tensor(X_train, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
X_val = torch.tensor(X_val, dtype=torch.float64)
y_val = torch.tensor(y_val, dtype=torch.float64)
X_test = torch.tensor(X_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

# Add bias column (ones) to all splits
X_train = torch.cat([torch.ones(X_train.shape[0], 1, dtype=X_train.dtype), X_train], dim=-1)
X_val = torch.cat([torch.ones(X_val.shape[0], 1, dtype=X_val.dtype), X_val], dim=-1)
X_test = torch.cat([torch.ones(X_test.shape[0], 1, dtype=X_test.dtype), X_test], dim=-1)

# Save in the required format
out = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}
torch.save(out, '../data/wecperth49.pt')
print('Saved higgs dataset to ../data/wecperth49.pt')
