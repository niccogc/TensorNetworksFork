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
grouped = df.groupby(['model_type', 'dataset'])['metric'].agg(['mean', 'sem']).reset_index()

# Create formatted strings with mean ± SEM
grouped['formatted'] = grouped['mean'].round(2).astype(str) + ' $\pm$ ' + grouped['sem'].round(2).astype(str)

#%%
# Create pivot table with mean values for ranking
pivot_mean = grouped.pivot(
    index="model_type",
    columns="dataset",
    values="mean"
)

# Create pivot table with formatted strings
pivot = grouped.pivot(
    index="model_type",   # rows
    columns="dataset",    # columns
    values="formatted"    # cell values with mean ± SEM
)

# Order by:
row_order = ['tt', 'tt_type1', 'tt_cumsum', 'tt_cumsum_type1', 'tt_lin', 'tt_lin_type1']
pivot = pivot.reindex(row_order)
pivot_mean = pivot_mean.reindex(row_order)
# Sort columns alphabetically
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
pivot_mean = pivot_mean.reindex(sorted(pivot_mean.columns), axis=1)
#%%
column_order = [
    'student_perf', 'abalone', 'obesity', 'bike', 'realstate', 'energy_efficiency', 'concrete', 'ai4i', 'popularity', 'seoulBike',
    'iris', 'hearth', 'winequalityc', 'breast', 'adult', 'bank', 'wine', 'car_evaluation', 'student_dropout', 'mushrooms'
]
pivot = pivot.reindex(columns=column_order)
pivot_mean = pivot_mean.reindex(columns=column_order)

#%%
# Apply formatting for best (bold) and second best (underline) entries
pivot_formatted = pivot.copy()

for col in pivot_mean.columns:
    if pivot_mean[col].notna().sum() > 0:  # Only process columns with data
        # Get rankings (higher is better)
        ranks = pivot_mean[col].rank(ascending=False, method='min')

        for idx in pivot_mean.index:
            if pd.notna(pivot_mean.loc[idx, col]):
                if ranks.loc[idx] == 1:  # Best
                    pivot_formatted.loc[idx, col] = f"\\textbf{{{pivot.loc[idx, col]}}}"
                elif ranks.loc[idx] == 2:  # Second best
                    pivot_formatted.loc[idx, col] = f"\\underline{{{pivot.loc[idx, col]}}}"

#%% Split into regression and classification tables
regression_pivot = pivot_formatted[[
    'student_perf', 'abalone', 'obesity', 'bike', 'realstate', 'energy_efficiency', 'concrete', 'ai4i', 'popularity', 'seoulBike'
]]

# Convert to LaTeX table for regression
latex_table_regression = regression_pivot.to_latex(na_rep="NA", caption="Model Ablation Performance Comparison - Regression (Mean ± SEM)", label="tab:model_ablation_regression", bold_rows=True, escape=False)
print(latex_table_regression.replace("_", " "))

classification_pivot = pivot_formatted[[
    'iris', 'hearth', 'winequalityc', 'breast', 'adult', 'bank', 'wine', 'car_evaluation', 'student_dropout', 'mushrooms'
]]

# Convert to LaTeX table for classification
latex_table_classification = classification_pivot.to_latex(na_rep="NA", caption="Model Ablation Performance Comparison - Classification (Mean ± SEM)", label="tab:model_ablation_classification", bold_rows=True, escape=False)
print(latex_table_classification.replace("_", " "))

# %%
# CSV text:
print("\nREGRESSION CSV:")
print(regression_pivot)
print("\nCLASSIFICATION CSV:")
print(classification_pivot)

# %%
