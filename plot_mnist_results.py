#%%
# Find csv's in current folder with names such as `tt_tt_N3_r3_ldNone_swipes20_P32970_fit_mnist.csv`

import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

files = glob.glob("*mnist.csv")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for file in files:
    df = pd.read_csv(file)
    # Split name into parts. For example:
    #`tt_tt_N3_r3_ldNone_swipes20_P32970_fit_mnist.csv`
    # should be split into
    # ['tt', 'N=3', 'r=3', 'ld=N/A', 'swipes=20', 'P=32970']
    # Every file starts with `tt' no matter the model type
    parts = file.split("_")[1:-2]
    name_parts = []
    for i, part in enumerate(parts):
        if part.startswith("N"):
            name_parts.append(" ".join(parts[:i]))
            name_parts.append(f"N={part[1:]}")
        elif part.startswith("r"):
            name_parts.append(f"r={part[1:]}")
        elif part.startswith("ld"):
            ld = part[2:]
            if ld == "None":
                ld = "N/A"
            name_parts.append(f"ld={ld}")
        elif part.startswith("swipes"):
            name_parts.append(f"swipes={part[6:]}")
        elif part.startswith("P"):
            name_parts.append(f"P={part[1:]}")
        elif part.startswith("cb"):
            # If it is -1 then print N/A
            if part[2:] == "-1":
                name_parts.append("cb=N/A")
            else:
                name_parts.append(f"cb={part[2:]}")
    name = ", ".join(name_parts)
    df['Model'] = name
    sns.lineplot(data=df, x='Epoch', y='Val Accuracy', ax=ax, label=name)
plt.ylim(0.8, 1.0)
plt.xlim(0, 80)
plt.tight_layout()
#%%