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
#%%
pivot = df.pivot(
    index="model_type",   # rows
    columns="dataset",    # columns
    values="metric"       # cell values
)

# Order by:
row_order = ['tt', 'tt_type1', 'tt_cumsum', 'tt_cumsum_type1', 'tt_lin', 'tt_lin_type1']
pivot = pivot.reindex(row_order)
# Sort columns alphabetically
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
#%%
# Convert to LaTeX table
latex_table = pivot.to_latex(float_format="%.2f", na_rep="NA", caption="Model Performance Comparison", label="tab:model_performance", bold_rows=True)
print(latex_table.replace("_", " "))
# %%
# CSV text:
print(pivot)

# %%
