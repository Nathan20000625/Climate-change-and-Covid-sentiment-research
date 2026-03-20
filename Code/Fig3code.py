import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ===== Path configuration =====
input_dir = r"E:\Figure\Fig3"
input_file = os.path.join(input_dir, "fig3-data.csv")

output_jpeg = os.path.join(input_dir, "Figure_3.jpeg")

# 1. Load and preprocess data
df3 = pd.read_csv(input_file)
df3['emotion'] = df3['emotion'].astype(str).str.capitalize()

# Enforce ordering: high-arousal negative first, then positive and cognitive emotions
emotions_order = ['Anger', 'Sadness', 'Fear', 'Disgust', 'Joy', 'Anticipation', 'Trust', 'Surprise']

matrix_data = []
annot_data = []

# 2. Loop through emotions and model directions, extract coefficients and filter by p-value
for emotion in emotions_order:
    row_data = []
    row_annot = []

    # Direction 1: COVID-19 predicting climate (forward)
    fw_subset = df3[(df3['emotion'] == emotion) & (df3['direction'] == 'forward')]
    if not fw_subset.empty:
        fw_row = fw_subset.iloc[0]

        # Short-run coefficient
        sr_p = fw_row['short_run_pvalue']
        sr_coef = fw_row['short_run_coefficient']
        sr_val = sr_coef if sr_p < 0.05 else 0.0
        sr_star = '***' if sr_p < 0.001 else '**' if sr_p < 0.01 else '*' if sr_p < 0.05 else ''
        row_data.append(sr_val)
        row_annot.append(f"{sr_val:.3f}\n{sr_star}" if sr_star else "")

        # Long-run coefficient
        lr_p = fw_row['long_run_pvalue']
        lr_coef = fw_row['long_run_coefficient']
        lr_val = lr_coef if lr_p < 0.05 else 0.0
        lr_star = '***' if lr_p < 0.001 else '**' if lr_p < 0.01 else '*' if lr_p < 0.05 else ''
        row_data.append(lr_val)
        row_annot.append(f"{lr_val:.3f}\n{lr_star}" if lr_star else "")
    else:
        row_data.extend([0.0, 0.0])
        row_annot.extend(["", ""])

    # Direction 2: climate predicting COVID-19 (reverse)
    rv_subset = df3[(df3['emotion'] == emotion) & (df3['direction'] == 'reverse')]
    if not rv_subset.empty:
        rv_row = rv_subset.iloc[0]

        # Short-run coefficient
        sr_p = rv_row['short_run_pvalue']
        sr_coef = rv_row['short_run_coefficient']
        sr_val = sr_coef if sr_p < 0.05 else 0.0
        sr_star = '***' if sr_p < 0.001 else '**' if sr_p < 0.01 else '*' if sr_p < 0.05 else ''
        row_data.append(sr_val)
        row_annot.append(f"{sr_val:.3f}\n{sr_star}" if sr_star else "")

        # Long-run coefficient
        lr_p = rv_row['long_run_pvalue']
        lr_coef = rv_row['long_run_coefficient']
        lr_val = lr_coef if lr_p < 0.05 else 0.0
        lr_star = '***' if lr_p < 0.001 else '**' if lr_p < 0.01 else '*' if lr_p < 0.05 else ''
        row_data.append(lr_val)
        row_annot.append(f"{lr_val:.3f}\n{lr_star}" if lr_star else "")
    else:
        row_data.extend([0.0, 0.0])
        row_annot.extend(["", ""])

    matrix_data.append(row_data)
    annot_data.append(row_annot)

# Define column names
columns = [
    'COVID-19 $\\rightarrow$ Climate\n(Short-run)',
    'COVID-19 $\\rightarrow$ Climate\n(Long-run)',
    'Climate $\\rightarrow$ COVID-19\n(Short-run)',
    'Climate $\\rightarrow$ COVID-19\n(Long-run)'
]

coef_df = pd.DataFrame(matrix_data, index=emotions_order, columns=columns)
annot_df = pd.DataFrame(annot_data, index=emotions_order, columns=columns)

# 3. Configure plotting
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
fig, ax = plt.subplots(figsize=(12, 8))

# Use white-to-red gradient colormap (highlighting positive regression coefficients)
cmap = sns.color_palette("Reds", as_cmap=True)

# Create mask: cells with value 0 are masked, revealing the light gray background
mask = coef_df == 0
ax.set_facecolor('#F0F0F0')

sns.heatmap(
    coef_df,
    annot=annot_df,
    fmt='',
    cmap=cmap,
    mask=mask,
    cbar_kws={'label': 'Mutual Regression Coefficient ($\\beta$)', 'shrink': 0.8},
    linewidths=2,
    linecolor='white',
    ax=ax,
    vmin=0,
    vmax=coef_df.max().max()
)

# Format fonts and labels
ax.xaxis.tick_top()
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold', rotation=0)

# 4. Add layout dividing lines (minimalist visual design)
ax.axvline(2, color='white', linewidth=6)
ax.axhline(4, color='white', linewidth=6)

# 5. Save figure (to E:\Figure\Fig3)
plt.tight_layout()
plt.savefig(
    output_jpeg,
    format='jpeg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    pil_kwargs={'quality': 95},
)
print(f"Export successful: {output_jpeg}")