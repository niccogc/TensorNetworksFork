#%%
from glob import glob
import pandas as pd
import numpy as np

# =========================
# 1) Load and aggregate
# =========================
paths = glob("./results/*.csv")

df_list = []
for p in paths:
    df_list.append(pd.read_csv(p))
df = pd.concat(df_list, ignore_index=True)

# Ensure numeric for metrics that may exist
for col in ["val_r2", "val_accuracy"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Determine task type and pick the metric
df["is_classification"] = df["val_accuracy"].notna() if "val_accuracy" in df.columns else False
df["metric"] = np.where(df["is_classification"], df.get("val_accuracy"), df.get("val_r2"))

# Group and compute mean and SEM per seed group
group_cols = [c for c in ["dataset", "N", "r", "lin_dim", "model_type"] if c in df.columns]
agg = (
    df.groupby(group_cols, dropna=False)
      .agg(
          n_seeds=("seed", "nunique") if "seed" in df.columns else ("metric", "size"),
          is_classification=("is_classification", "any"),
          metric_mean=("metric", "mean"),
          metric_sem=("metric", "sem"),
          num_params=("num_params", "first") if "num_params" in df.columns else ("metric", "size")
      )
      .reset_index()
      .sort_values(group_cols)
)

# Pull one num_params per non-dataset identifier group
key_cols = [c for c in group_cols if c != "dataset"]
params = (
    agg.groupby(key_cols, dropna=False)["num_params"]
       .first()
       .reset_index()
)

# Pivot mean and SEM so each dataset is a column
pivoted = agg.pivot(index=key_cols, columns="dataset", values="metric_mean").reset_index()
pivoted_sem = agg.pivot(index=key_cols, columns="dataset", values="metric_sem").reset_index()

# Merge num_params back
pivoted = pivoted.merge(params, on=key_cols, how="left")

# Optional ordering and sorting
lead = [c for c in ["model_type", "N", "r", "lin_dim", "num_params"] if c in pivoted.columns]
rest = [c for c in pivoted.columns if c not in lead]
pivoted = pivoted[lead + rest]

# If you have a desired order for model types
if "model_type" in pivoted.columns:
    model_order = [
        "tt", "tt_type1", "tt_lin", "tt_lin_type1",
        "tt_cumsum", "tt_type1_cumsum", "cpd", "cpd_type1"
    ]
    pivoted["model_type"] = pd.Categorical(
        pivoted["model_type"],
        categories=model_order,
        ordered=True
    )

sort_by = [c for c in ["model_type", "N", "r", "lin_dim"] if c in pivoted.columns]
if sort_by:
    pivoted = pivoted.sort_values(by=sort_by, ascending=[True]*len(sort_by)).reset_index(drop=True)

# =========================
# 2) Build formatted table
# =========================
# Identify dataset columns from mean table
dataset_cols = [c for c in pivoted.columns if c not in [*key_cols, "num_params"]]

# Numeric copies for ranking
formatted = pivoted.copy()
for c in dataset_cols:
    formatted[c] = pd.to_numeric(formatted[c], errors="coerce")

# Numeric SEM table aligned to mean columns
sem_formatted = pivoted_sem.copy()
for c in dataset_cols:
    if c in sem_formatted.columns:
        sem_formatted[c] = pd.to_numeric(sem_formatted[c], errors="coerce")
    else:
        sem_formatted[c] = np.nan

# String table to display
str_df = pivoted.copy()

pd.set_option('display.max_columns', None)  # Show all columns
print("Full pivoted table:")
print(pivoted.to_string(index=False))
#%%
# Compose "mean ± sem" strings, using "--" when mean is NaN
def fmt_mean_sem(m, s):
    if pd.isna(m):
        return "--"
    if pd.isna(s):
        return f"{m*100:.1f}"
    return f"{m*100:.1f} ± {s*100:.1f}"

for c in dataset_cols:
    m = formatted[c]
    s = sem_formatted[c]
    str_df[c] = [fmt_mean_sem(m.iat[i], s.iat[i]) for i in range(len(str_df))]

# Replace underscores in model_type names for display
if "model_type" in str_df.columns:
    str_df["model_type"] = str_df["model_type"].astype(str).str.replace("_", " ")

# Replace NaNs in key columns such as lin_dim with "--"
for kc in [k for k in ["model_type", "N", "r", "lin_dim", "num_params"] if k in str_df.columns]:
    mask_nan = pivoted[kc].isna()
    if mask_nan.any():
        # Keep existing values where not NaN, set string "--" where NaN
        str_df[kc] = str_df[kc].astype(object)
        str_df.loc[mask_nan, kc] = "--"

mask_numeric = pd.to_numeric(pivoted["lin_dim"], errors="coerce").notna()
str_df.loc[mask_numeric, "lin_dim"] = pivoted.loc[mask_numeric, "lin_dim"].map(lambda x: f"{x:.2f}")

# =========================
# 3) Rank within each model_type and dataset
# =========================
# Determine model types present, respecting categorical order when available
if "model_type" in formatted.columns and pd.api.types.is_categorical_dtype(formatted["model_type"]):
    model_types = formatted["model_type"].cat.categories.tolist()
else:
    model_types = formatted["model_type"].dropna().unique().tolist() if "model_type" in formatted.columns else ["_all_"]

# For each model_type and each dataset column, bold the best and underline the second best
for mt in model_types:
    if "model_type" in formatted.columns:
        mask = formatted["model_type"] == mt
    else:
        mask = pd.Series(True, index=formatted.index)

    if not mask.any():
        continue

    for col in dataset_cols:
        # Rank by numeric mean
        series = formatted.loc[mask, col].dropna()
        if series.empty:
            continue

        # Higher is better for both accuracy and R2
        sorted_idx = series.sort_values(ascending=False).index.to_list()

        best_idx = sorted_idx[0]
        if str_df.at[best_idx, col] not in ["", "--"]:
            str_df.at[best_idx, col] = r"\textbf{" + str_df.at[best_idx, col] + "}"

        # if len(sorted_idx) > 1:
        #     second_idx = sorted_idx[1]
        #     if str_df.at[second_idx, col] not in ["", "--"]:
        #         str_df.at[second_idx, col] = r"\underline{" + str_df.at[second_idx, col] + "}"

# =========================
# 4) Export LaTeX
# =========================
str_df.columns = str_df.columns.str.replace("_", " ", regex=False)
latex_table = str_df.to_latex(index=False, escape=False)
print(latex_table)

#%%
# %%
# =========================
# Extract bolded entries and compare across model types
# =========================

# Safety checks for objects created above
assert "formatted" in globals() and "sem_formatted" in globals(), "Run the previous cell first."
assert "dataset_cols" in globals(), "dataset_cols must be defined from the previous cell."

mt_col = "model_type" if "model_type" in formatted.columns else None

# Determine model types, preserving categorical order if available
if mt_col and pd.api.types.is_categorical_dtype(formatted[mt_col]):
    model_types_iter = [c for c in formatted[mt_col].cat.categories if c in set(formatted[mt_col].dropna())]
elif mt_col:
    model_types_iter = formatted[mt_col].dropna().unique().tolist()
else:
    model_types_iter = ["_all_"]  # Fallback if there is no model_type column

best_rows = []
for mt in model_types_iter:
    mask = formatted[mt_col] == mt if mt_col else pd.Series(True, index=formatted.index)
    if not mask.any():
        continue

    for ds in dataset_cols:
        series = formatted.loc[mask, ds]
        if series.notna().any():
            # Index of the best entry within this model_type for this dataset
            idx = series.idxmax()
            mean = formatted.at[idx, ds]
            sem = sem_formatted.at[idx, ds] if ds in sem_formatted.columns else np.nan

            row = {
                "model_type": str(formatted.at[idx, mt_col]) if mt_col else "_all_",
                "dataset": ds,
                "mean": mean,
                "sem": sem,
            }

            # Optionally keep the winning configuration identifiers
            for kc in [k for k in ["N", "r", "lin_dim", "num_params"] if k in formatted.columns]:
                row[kc] = formatted.at[idx, kc]

            best_rows.append(row)

# Build a tidy table of the bolded entries
best_df = pd.DataFrame(best_rows)

# Wide numeric mean and sem
best_mean_wide = best_df.pivot(index="model_type", columns="dataset", values="mean")
best_sem_wide  = best_df.pivot(index="model_type", columns="dataset", values="sem")

# Compose "mean ± sem" strings for comparison across model types
def _fmt(m, s):
    if pd.isna(m):
        return "--"
    if pd.isna(s):
        return f"{m*100:.1f}"
    return f"{m*100:.1f} ± {s*100:.1f}"

comparison = best_mean_wide.copy()
for ds in comparison.columns:
    mcol = best_mean_wide[ds]
    scol = best_sem_wide[ds] if ds in best_sem_wide.columns else pd.Series(np.nan, index=comparison.index)
    comparison[ds] = [_fmt(mcol.iat[i], scol.iat[i]) for i in range(len(comparison))]

# Optional: attach the winning configuration for each dataset as separate columns
config_cols = [k for k in ["N", "r", "lin_dim", "num_params"] if k in best_df.columns]
if config_cols:
    cfg = (
        best_df.assign(
            config=lambda d: d[config_cols].astype(str).agg(", ".join, axis=1)
        )
        .pivot(index="model_type", columns="dataset", values="config")
        .add_suffix(" (config)")
    )
    comparison = comparison.join(cfg, how="left")

# Clean column names for LaTeX if you wish to export
latex_ready = comparison.copy()
latex_ready.columns = latex_ready.columns.astype(str).str.replace("_", " ", regex=False)

print("Best per model type and dataset, extracted from the bolded entries:")
print(comparison.reset_index().to_string(index=False))
#%%
# Split into regression and classification for better readability
# Then sort by metric_mean descending and print all rows
regression_df = agg[~agg["is_classification"]].drop(columns=["is_classification"]).sort_values("metric_mean", ascending=False)
classification_df = agg[agg["is_classification"]].drop(columns=["is_classification"]).sort_values("metric_mean", ascending=False)
pd.set_option('display.max_rows', None)  # Show all rows
print("=== Regression Results ===")
print(regression_df.to_string(index=False))
print("\n=== Classification Results ===")
print(classification_df.to_string(index=False))
# %%
