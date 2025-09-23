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
df = df[['dataset', 'model_type', 'num_params', 'metric']]

# Calculate mean and SEM for each model-dataset combination
import numpy as np
grouped = df.groupby(['model_type', 'dataset'])['metric'].agg(['mean', 'sem']).reset_index()

# Create formatted strings with mean ± SEM
grouped['formatted'] = grouped['mean'].round(2).astype(str) + ' $\pm$ ' + grouped['sem'].round(2).astype(str)

#%%
pivot = grouped.pivot(
    index="model_type",   # rows
    columns="dataset",    # columns
    values="formatted"    # cell values with mean ± SEM
)

# Order by:
row_order = ['tt', 'tt_type1', 'tt_cumsum', 'tt_cumsum_type1', 'tt_lin', 'tt_lin_type1']
pivot = pivot.reindex(row_order)
# Sort columns alphabetically
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
#%%
column_order = [
    'student_perf', 'abalone', 'obesity', 'bike', 'realstate', 'energy_efficiency', 'concrete', 'ai4i', 'popularity', 'seoulBike',
    'iris', 'hearth', 'winequalityc', 'breast', 'adult', 'bank', 'wine', 'car_evaluation', 'student_dropout', 'mushrooms'
]
pivot = pivot.reindex(columns=column_order)
#%%
# Convert to LaTeX table
latex_table = pivot.to_latex(na_rep="NA", caption="Model Performance Comparison (Mean ± SEM)", label="tab:model_performance", bold_rows=True, escape=False)
print(latex_table.replace("_", " "))
# %%
# CSV text:
print(pivot)

# %%
