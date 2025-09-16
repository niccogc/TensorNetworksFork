#%%
from glob import glob
import pandas as pd

import pandas as pd

col_names = [
    "timestamp",
    "model_type",
    "dataset",
    "N",
    "r",
    "lin_dim",
    "test_rmse",
    "test_r2",
    "test_accuracy",
    "num_params",
    "converged_epoch"
]

paths = glob("./results/test_results_tt*.csv")

df_list = []
for p in paths:
    df_list.append(pd.read_csv(p, header=None, names=col_names))
df = pd.concat(df_list, ignore_index=True)

df['metric'] = df['test_accuracy'].where(df['test_accuracy'].notna(), df['test_r2']) * 100
#%%
best_per_dataset = df.loc[df.groupby('dataset')['metric'].idxmax()].reset_index(drop=True)
#%%
# 1) Add a model flag to your existing best rows
best_per_dataset['model'] = 'tt'  # your TT runs
df_tt = best_per_dataset[['dataset', 'model', 'model_type', 'num_params', 'metric']].copy()

# 2) Read the CPD summary file and drop SEM columns
cpd_sym_type1_raw = pd.read_csv('./results/matlab_results_20250915_104118.csv')

# 3) Build the CPD metric: accuracy when present, else R2
cpd_sym_type1_metric = cpd_sym_type1_raw['Mean_Test_Accuracy'].where(
    cpd_sym_type1_raw['Mean_Test_Accuracy'].notna(),
    cpd_sym_type1_raw['Mean_Test_R2']*100
)

# 4) Put CPD data into the same schema
df_cpd_type1 = pd.DataFrame({
    'dataset': cpd_sym_type1_raw['Dataset'],
    'model': 'cpd_sym_type1',
    'model_type': 'cpd_sym_type1',
    'num_params': pd.NA,     # not provided in the summary file
    'metric': cpd_sym_type1_metric
})

# 2) Read the CPD summary file and drop SEM columns
cpd_sym_raw = pd.read_csv('./results/matlab_results_type2_20250915_111937.csv')

# 3) Build the CPD metric: accuracy when present, else R2
cpd_sym_metric = cpd_sym_raw['Mean_Test_Accuracy'].where(
    cpd_sym_raw['Mean_Test_Accuracy'].notna(),
    cpd_sym_raw['Mean_Test_R2']*100
)

# 4) Put CPD data into the same schema
df_cpd = pd.DataFrame({
    'dataset': cpd_sym_raw['Dataset'],
    'model': 'cpd_sym',
    'model_type': 'cpd_sym',
    'num_params': pd.NA,     # not provided in the summary file
    'metric': cpd_sym_metric
})

#%%
paths = ["./results/test_results_cpd.csv", "./results/test_results_cpd_type1.csv"]
models = ['cpd', 'cpd_type1']

df_list = []
for p, m in zip(paths, models):
    df = pd.read_csv(p, header=None, names=col_names)
    df['model'] = m
    df['metric'] = df['test_accuracy'].where(df['test_accuracy'].notna(), df['test_r2']) * 100
    df_list.append(df[['dataset', 'model', 'model_type', 'num_params', 'metric']])
df_cpd_combined = pd.concat(df_list, ignore_index=True)
#%%

# 5) Concatenate
combined = pd.concat([df_tt, df_cpd, df_cpd_type1, df_cpd_combined], ignore_index=True)
#%%
pivot = combined.pivot(
    index="model",   # rows
    columns="dataset",    # columns
    values="metric"       # cell values
)
# %%
