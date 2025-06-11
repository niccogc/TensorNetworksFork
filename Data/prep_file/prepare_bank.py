from ucimlrepo import fetch_ucirepo 
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# fetch dataset 
bank = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank.data.features 
y = bank.data.targets 
cat_columns = [
    'month',
    'default',
    'poutcome',
    'education',
    'marital',
    'job',
    'contact',
    'housing',
    'loan'
]

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
# Standardize features
scaler = StandardScaler()
# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False)
Xcat_encoded = encoder.fit_transform(X[cat_columns])

# Drop categorical columns and scale numerical ones
Xnum = X.drop(columns=cat_columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xnum)

# Concatenate scaled numeric and one-hot encoded categorical
X_scaled = np.hstack([X_scaled, Xcat_encoded])

# Split into train, val, test (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_onehot, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

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
torch.save(out, '../data/bank_tensor.pt')
print('Saved bank dataset to ../data/bank_tensor.pt')
