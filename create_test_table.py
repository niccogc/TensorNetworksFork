#%%
from glob import glob
import pandas as pd
import numpy as np

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

# For TT models: group by dataset and model_type, find best model_type per dataset
# Calculate mean metric for each dataset-model_type combination
tt_grouped = df.groupby(['dataset', 'model_type'])['metric'].agg(['mean', 'sem', 'count']).reset_index()
# Find best model_type per dataset (highest mean metric)
best_model_types = tt_grouped.loc[tt_grouped.groupby('dataset')['mean'].idxmax()].reset_index(drop=True)

# Now get all runs for the best model_type per dataset and calculate final mean/SEM
tt_best_runs = []
for _, row in best_model_types.iterrows():
    dataset = row['dataset']
    model_type = row['model_type']
    dataset_model_runs = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
    mean_metric = dataset_model_runs['metric'].mean()
    sem_metric = dataset_model_runs['metric'].sem()
    mean_params = dataset_model_runs['num_params'].mean()

    tt_best_runs.append({
        'dataset': dataset,
        'model': 'tt',
        'model_type': model_type,
        'num_params': mean_params,
        'metric_mean': mean_metric,
        'metric_sem': sem_metric,
        'metric': f"{mean_metric:.2f} ± {sem_metric:.2f}" if not np.isnan(sem_metric) else f"{mean_metric:.2f}"
    })

df_tt = pd.DataFrame(tt_best_runs)

# 2) Read the CPD summary file and drop SEM columns
cpd_sym_type1_raw = pd.read_csv('./results/matlab_results_20250915_104118.csv')

# 3) Build the CPD metric: accuracy when present, else R2
cpd_sym_type1_metric = cpd_sym_type1_raw['Mean_Test_Accuracy'].where(
    cpd_sym_type1_raw['Mean_Test_Accuracy'].notna(),
    cpd_sym_type1_raw['Mean_Test_R2']*100
)

#%%
# Load CPD and CPD Type1 data together (raw individual runs)
paths = ["./results/test_results_cpd.csv", "./results/test_results_cpd_type1.csv"]
models = ['cpd', 'cpd_type1']

all_cpd_data = []
for p, m in zip(paths, models):
    df = pd.read_csv(p, header=None, names=col_names)
    df['model'] = m
    df['metric'] = df['test_accuracy'].where(df['test_accuracy'].notna(), df['test_r2']) * 100
    all_cpd_data.append(df)

# Combine all CPD data (both cpd and cpd_type1)
df_all_cpd = pd.concat(all_cpd_data, ignore_index=True)

# Group by dataset, model, and model_type to find best combination per dataset
cpd_grouped = df_all_cpd.groupby(['dataset', 'model', 'model_type'])['metric'].agg(['mean', 'sem', 'count']).reset_index()
# Find best model+model_type combination per dataset (highest mean metric)
best_cpd_combinations = cpd_grouped.loc[cpd_grouped.groupby('dataset')['mean'].idxmax()].reset_index(drop=True)

# Calculate final mean and SEM for best combination per dataset
cpd_best_runs = []
for _, row in best_cpd_combinations.iterrows():
    dataset = row['dataset']
    model = row['model']
    model_type = row['model_type']
    dataset_model_runs = df_all_cpd[(df_all_cpd['dataset'] == dataset) &
                                    (df_all_cpd['model'] == model) &
                                    (df_all_cpd['model_type'] == model_type)]
    mean_metric = dataset_model_runs['metric'].mean()
    sem_metric = dataset_model_runs['metric'].sem()
    mean_params = dataset_model_runs['num_params'].mean()

    cpd_best_runs.append({
        'dataset': dataset,
        'model': 'cpd',  # Unified model name for the table
        'model_type': f"{model}_{model_type}",  # Keep track of which was best
        'num_params': mean_params,
        'metric_mean': mean_metric,
        'metric_sem': sem_metric,
        'metric': f"{mean_metric:.2f} ± {sem_metric:.2f}" if not np.isnan(sem_metric) else f"{mean_metric:.2f}"
    })

df_cpd_combined = pd.DataFrame(cpd_best_runs)

#%%
# Process CPD Sym data (already has mean and SEM calculated)
# CPD Sym Type I
cpd_sym_type1_metric_mean = cpd_sym_type1_raw['Mean_Test_Accuracy'].where(
    cpd_sym_type1_raw['Mean_Test_Accuracy'].notna(),
    cpd_sym_type1_raw['Mean_Test_R2']*100
)
cpd_sym_type1_metric_sem = cpd_sym_type1_raw['SEM_Test_Accuracy'].where(
    cpd_sym_type1_raw['SEM_Test_Accuracy'].notna(),
    cpd_sym_type1_raw['SEM_Test_R2']*100
)

df_cpd_sym_type1 = pd.DataFrame({
    'dataset': cpd_sym_type1_raw['Dataset'],
    'model': 'cpd_sym_type1',
    'metric_mean': cpd_sym_type1_metric_mean,
    'metric_sem': cpd_sym_type1_metric_sem
})

# CPD Sym
cpd_sym_raw = pd.read_csv('./results/matlab_results_type2_20250915_111937.csv')
cpd_sym_metric_mean = cpd_sym_raw['Mean_Test_Accuracy'].where(
    cpd_sym_raw['Mean_Test_Accuracy'].notna(),
    cpd_sym_raw['Mean_Test_R2']*100
)
cpd_sym_metric_sem = cpd_sym_raw['SEM_Test_Accuracy'].where(
    cpd_sym_raw['SEM_Test_Accuracy'].notna(),
    cpd_sym_raw['SEM_Test_R2']*100
)

df_cpd_sym = pd.DataFrame({
    'dataset': cpd_sym_raw['Dataset'],
    'model': 'cpd_sym',
    'metric_mean': cpd_sym_metric_mean,
    'metric_sem': cpd_sym_metric_sem
})

# Combine CPD Sym variants and find best per dataset
df_all_cpd_sym = pd.concat([df_cpd_sym, df_cpd_sym_type1], ignore_index=True)
best_cpd_sym = df_all_cpd_sym.loc[df_all_cpd_sym.groupby('dataset')['metric_mean'].idxmax()].reset_index(drop=True)

# Format the CPD Sym results
cpd_sym_final = []
for _, row in best_cpd_sym.iterrows():
    mean_val = row['metric_mean']
    sem_val = row['metric_sem']
    cpd_sym_final.append({
        'dataset': row['dataset'],
        'model': 'cpd_sym',  # Unified model name for the table
        'model_type': row['model'],  # Keep track of which was best
        'num_params': pd.NA,
        'metric_mean': mean_val,
        'metric_sem': sem_val,
        'metric': f"{mean_val:.2f} ± {sem_val:.2f}" if not np.isnan(sem_val) else f"{mean_val:.2f}"
    })

df_cpd_sym_combined = pd.DataFrame(cpd_sym_final)
#%%
#%%
paths = ["./results/test_results_mlp.csv", "./results/test_results_tnml_polynomial.csv", "./results/test_results_tnml_sin-cos.csv", "./results/test_results_xgboost.csv", "./results/test_results_gp.csv"]
models = ['mlp', 'tnml_polynomial', 'tnml_sin-cos', 'xgboost', 'gp']
# tnml csv header creater:
# f"{timestamp},{args.model_type},{dataset},{args.N},{args.r},{np.nan},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n"
# xgboost csv header creater:
# f.write(
#     f"{timestamp},{args.model_type},{dataset},{args.n_estimators},"
#     f"{args.max_depth},{result['test_rmse']},{result['test_r2']},"
#     f"{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n"
# )
# mlp csv header creater:
# f.write(f"{timestamp},{args.model_type},{dataset},{args.num_layers},{args.num_channels},{result['test_rmse']},{result['test_r2']},{result['test_accuracy']},{result['num_params']},{result['converged_epoch']}\n")

tnml_cols = ["timestamp","model_type","dataset","N","r","lin_dim","test_rmse","test_r2","test_accuracy","num_params","converged_epoch"]
xgboost_cols = ["timestamp","model_type","dataset","n_estimators","max_depth","test_rmse","test_r2","test_accuracy","num_params","converged_epoch"]
mlp_cols = ["timestamp","model_type","dataset","num_layers","num_channels","test_rmse","test_r2","test_accuracy","num_params","converged_epoch"]
gp_cols = ["timestamp","model_type","dataset","best_kernel_name","best_alpha","test_rmse","test_r2","test_accuracy","num_params","converged_epoch"]
col_names_list = [mlp_cols, tnml_cols, tnml_cols, xgboost_cols, gp_cols]

df_other_list = []
for p, m, cols in zip(paths, models, col_names_list):
    df = pd.read_csv(p, header=None, names=cols)
    df['model'] = m
    df['metric'] = df['test_accuracy'].where(df['test_accuracy'].notna(), df['test_r2']) * 100

    # For other models: group by dataset and model_type, find best performing combination
    # First, group by dataset and model_type to get mean performance
    other_grouped = df.groupby(['dataset', 'model_type'])['metric'].agg(['mean', 'sem', 'count']).reset_index()
    # Find best model_type per dataset
    best_other_types = other_grouped.loc[other_grouped.groupby('dataset')['mean'].idxmax()].reset_index(drop=True)

    # Calculate mean and SEM for best model_type per dataset
    other_best_runs = []
    for _, row in best_other_types.iterrows():
        dataset = row['dataset']
        model_type = row['model_type']
        dataset_model_runs = df[(df['dataset'] == dataset) & (df['model_type'] == model_type)]
        mean_metric = dataset_model_runs['metric'].mean()
        sem_metric = dataset_model_runs['metric'].sem()
        mean_params = dataset_model_runs['num_params'].mean()

        other_best_runs.append({
            'dataset': dataset,
            'model': m,
            'model_type': model_type,
            'num_params': mean_params,
            'metric_mean': mean_metric,
            'metric_sem': sem_metric,
            'metric': f"{mean_metric:.2f} ± {sem_metric:.2f}" if not np.isnan(sem_metric) else f"{mean_metric:.2f}"
        })

    df_other_list.append(pd.DataFrame(other_best_runs))

df_other_combined = pd.concat(df_other_list, ignore_index=True)
#%%
# 5) Concatenate
combined = pd.concat([df_tt, df_cpd_sym_combined, df_cpd_combined, df_other_combined], ignore_index=True)

# Create separate pivots for mean and SEM
pivot_mean = combined.pivot(
    index="model",
    columns="dataset",
    values="metric_mean"
)

pivot_sem = combined.pivot(
    index="model",
    columns="dataset",
    values="metric_sem"
)

# Find best and second best models for formatting
best_models = {}
second_best_models = {}

for dataset in pivot_mean.columns:
    dataset_values = pivot_mean[dataset].dropna()
    if len(dataset_values) >= 2:
        sorted_values = dataset_values.sort_values(ascending=False)
        best_models[dataset] = sorted_values.index[0]
        second_best_models[dataset] = sorted_values.index[1]
    elif len(dataset_values) == 1:
        best_models[dataset] = dataset_values.index[0]

# Create alternating mean/SEM table
alternating_rows = []
model_order = ['tt', 'cpd', 'cpd_sym', 'tnml_polynomial', 'tnml_sin-cos', 'gp', 'mlp', 'xgboost']

for model in model_order:
    if model in pivot_mean.index:
        # Mean row
        mean_row_data = {}
        for dataset in pivot_mean.columns:
            mean_val = pivot_mean.loc[model, dataset]
            if not pd.isna(mean_val):
                formatted_val = f"{mean_val:.2f}"
                # Apply formatting for best/second best
                if dataset in best_models and best_models[dataset] == model:
                    formatted_val = f"\\textbf{{{formatted_val}}}"
                elif dataset in second_best_models and second_best_models[dataset] == model:
                    formatted_val = f"\\underline{{{formatted_val}}}"
                mean_row_data[dataset] = formatted_val
            else:
                mean_row_data[dataset] = "NA"

        # SEM row
        sem_row_data = {}
        for dataset in pivot_sem.columns:
            sem_val = pivot_sem.loc[model, dataset]
            if not pd.isna(sem_val):
                sem_row_data[dataset] = f"({sem_val:.2f})"
            else:
                sem_row_data[dataset] = "NA"

        # Add both rows to list
        alternating_rows.append((f"{model}", mean_row_data))
        alternating_rows.append((f"{model}_sem", sem_row_data))

# Create the alternating DataFrame
alternating_data = {}
row_names = []

for row_name, row_data in alternating_rows:
    row_names.append(row_name)
    for dataset in row_data:
        if dataset not in alternating_data:
            alternating_data[dataset] = []
        alternating_data[dataset].append(row_data[dataset])

pivot = pd.DataFrame(alternating_data, index=row_names)
# Sort columns alphabetically
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
#%%
#%%
# Sort columns in this order:
# | ID  | Dataset            | Task           | Train | Val  | Test | Features |
# |-----|--------------------|----------------|-------|------|------|----------|
# | 320 | student perf       | regression     | 454   | 97   | 98   | 73       |
# | 1   | abalone            | regression     | 2923  | 627  | 627  | 11       |
# | 544 | obesity            | regression     | 1477  | 317  | 317  | 39       |
# | 275 | bike               | regression     | 12165 | 2607 | 2607 | 12       |
# | 477 | realstate          | regression     | 289   | 62   | 63   | 6        |
# | 242 | energy efficiency  | regression     | 537   | 115  | 116  | 8        |
# | 165 | concrete           | regression     | 721   | 154  | 155  | 8        |
# | 601 | ai4i               | regression     | 7000  | 1500 | 1500 | 9        |
# | 332 | popularity         | regression     | 27750 | 5947 | 5947 | 58       |
# | 560 | seoulBike          | regression     | 13903 | 2996 | 2996 | 14       |
# | 53  | iris               | classification | 105   | 22   | 23   | 4        |
# | 45  | hearth             | classification | 212   | 45   | 46   | 11       |
# | 186 | winequalityc       | classification | 4547  | 975  | 975  | 11       |
# | 17  | breast             | classification | 398   | 85   | 86   | 30       |
# | 2   | adult              | classification | 34189 | 7326 | 7327 | 47       |
# | 222 | bank               | classification | 31647 | 6782 | 6782 | 33       |
# | 109 | wine               | classification | 124   | 27   | 27   | 13       |
# | 19  | car evaluation     | classification | 1209  | 259  | 260  | 27       |
# | 697 | student dropout    | classification | 3096  | 664  | 664  | 36       |
# | 73  | mushrooms          | classification | 5686  | 1219 | 1219 | 100      |
# Sort dataset columns in this listed order:
column_order = [
    'student_perf', 'abalone', 'obesity', 'bike', 'realstate', 'energy_efficiency', 'concrete', 'ai4i', 'popularity', 'seoulBike',
    'iris', 'hearth', 'winequalityc', 'breast', 'adult', 'bank', 'wine', 'car_evaluation', 'student_dropout', 'mushrooms'
]
pivot = pivot.reindex(columns=column_order)
#%% Split into regression and classification tables and add average columns
regression_datasets = [
    'student_perf', 'abalone', 'obesity', 'bike', 'realstate', 'energy_efficiency', 'concrete', 'ai4i', 'popularity', 'seoulBike'
]
classification_datasets = [
    'iris', 'hearth', 'winequalityc', 'breast', 'adult', 'bank', 'wine', 'car_evaluation', 'student_dropout', 'mushrooms'
]

regression_pivot = pivot[regression_datasets].copy()
classification_pivot = pivot[classification_datasets].copy()

# Calculate averages for regression table and find best/second best
regression_avg_values = {}
for model in model_order:
    if model in pivot_mean.index:
        # Get mean values for regression datasets only
        reg_values = []
        for dataset in regression_datasets:
            if dataset in pivot_mean.columns:
                val = pivot_mean.loc[model, dataset]
                if not pd.isna(val):
                    reg_values.append(val)

        if reg_values:
            avg_val = np.mean(reg_values)
            regression_avg_values[model] = avg_val

# Find best and second best for regression averages
reg_sorted = sorted(regression_avg_values.items(), key=lambda x: x[1], reverse=True)
reg_best = reg_sorted[0][0] if len(reg_sorted) > 0 else None
reg_second_best = reg_sorted[1][0] if len(reg_sorted) > 1 else None

# Format regression averages
regression_averages = []
for model in model_order:
    if model in pivot_mean.index:
        if model in regression_avg_values:
            avg_val = regression_avg_values[model]
            formatted_val = f"{avg_val:.2f}"
            if model == reg_best:
                formatted_val = f"\\textbf{{{formatted_val}}}"
            elif model == reg_second_best:
                formatted_val = f"\\underline{{{formatted_val}}}"
            regression_averages.append(formatted_val)
        else:
            regression_averages.append("NA")

        # Add SEM row (empty for average column)
        regression_averages.append("")

# Calculate averages for classification table and find best/second best
classification_avg_values = {}
for model in model_order:
    if model in pivot_mean.index:
        # Get mean values for classification datasets only
        class_values = []
        for dataset in classification_datasets:
            if dataset in pivot_mean.columns:
                val = pivot_mean.loc[model, dataset]
                if not pd.isna(val):
                    class_values.append(val)

        if class_values:
            avg_val = np.mean(class_values)
            classification_avg_values[model] = avg_val

# Find best and second best for classification averages
class_sorted = sorted(classification_avg_values.items(), key=lambda x: x[1], reverse=True)
class_best = class_sorted[0][0] if len(class_sorted) > 0 else None
class_second_best = class_sorted[1][0] if len(class_sorted) > 1 else None

# Format classification averages
classification_averages = []
for model in model_order:
    if model in pivot_mean.index:
        if model in classification_avg_values:
            avg_val = classification_avg_values[model]
            formatted_val = f"{avg_val:.2f}"
            if model == class_best:
                formatted_val = f"\\textbf{{{formatted_val}}}"
            elif model == class_second_best:
                formatted_val = f"\\underline{{{formatted_val}}}"
            classification_averages.append(formatted_val)
        else:
            classification_averages.append("NA")

        # Add SEM row (empty for average column)
        classification_averages.append("")

# Add average columns to the pivots
regression_pivot['Average'] = regression_averages
classification_pivot['Average'] = classification_averages

# Convert to LaTeX table (no float format since we have formatted strings)
latex_table = regression_pivot.to_latex(na_rep="NA", caption="Model Performance Comparison", label="tab:model_performance", bold_rows=True, escape=False)
print(latex_table.replace("_", " "))
# Convert to LaTeX table (no float format since we have formatted strings)
latex_table = classification_pivot.to_latex(na_rep="NA", caption="Model Performance Comparison", label="tab:model_performance", bold_rows=True, escape=False)
print(latex_table.replace("_", " "))
# %%
# CSV text:
print(regression_pivot)

# CSV text:
print(classification_pivot)

# %%
