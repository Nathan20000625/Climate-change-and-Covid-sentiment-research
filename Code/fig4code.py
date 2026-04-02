import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_dirs = [
    _script_dir,
    r"E:\Figure\Fig4",
    r"F:\Figure\Fig4",
]


def _resolve_fig4_csv() -> str:
    env_path = os.environ.get("FIG4_DATA_CSV", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    for d in _candidate_dirs:
        p = os.path.join(d, "fig4-data.csv")
        if os.path.isfile(p):
            return p
    searched = [os.path.join(d, "fig4-data.csv") for d in _candidate_dirs]
    raise FileNotFoundError(
        "Cannot find fig4-data.csv. Copy the generated fig4-data.csv to this script directory, "
        "or set environment variable FIG4_DATA_CSV=<full_path>.\n"
        "(Default output from main analysis: .../analysis_results/fig4-data.csv)\n"
        "Searched paths:\n  " + "\n  ".join(searched)
    )


input_file = _resolve_fig4_csv()
input_dir = os.path.dirname(os.path.abspath(input_file))
output_jpeg = os.path.join(_script_dir, "Figure_4.jpeg")


def _short_long_col_names(df: pd.DataFrame):
    """Support both export_fig4_data (*_coeff) and legacy stacked (*_coefficient) column names."""
    if "short_run_coeff" in df.columns:
        c_short = "short_run_coeff"
    elif "short_run_coefficient" in df.columns:
        c_short = "short_run_coefficient"
    else:
        raise KeyError("CSV is missing short_run_coeff or short_run_coefficient")
    if "long_run_coeff" in df.columns:
        c_long = "long_run_coeff"
    elif "long_run_coefficient" in df.columns:
        c_long = "long_run_coefficient"
    else:
        raise KeyError("CSV is missing long_run_coeff or long_run_coefficient")
    return c_short, c_long


# 1. Load and preprocess data
df3 = pd.read_csv(input_file, encoding="utf-8-sig")
df3["emotion"] = df3["emotion"].astype(str).str.capitalize()
COL_SHORT, COL_LONG = _short_long_col_names(df3)

# Enforce ordering: high-arousal negative first, then positive and cognitive emotions
emotions_order = ["Anger", "Sadness", "Fear", "Disgust", "Joy", "Anticipation", "Trust", "Surprise"]

matrix_data = []
annot_data = []


def _sig_stars(p) -> str:
    try:
        if pd.isna(p):
            return ""
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _cell_value(coef, p) -> float:
    try:
        if pd.isna(p) or float(p) >= 0.05:
            return 0.0
        c = float(coef)
        return c if np.isfinite(c) else 0.0
    except (TypeError, ValueError):
        return 0.0


# 2. Loop through emotions and model directions, extract coefficients and filter by p-value
for emotion in emotions_order:
    row_data = []
    row_annot = []

    # Direction 1: COVID-19 predicting climate (forward)
    fw_subset = df3[(df3["emotion"] == emotion) & (df3["direction"] == "forward")]
    if not fw_subset.empty:
        fw_row = fw_subset.iloc[0]

        sr_p = fw_row["short_run_pvalue"]
        sr_coef = fw_row[COL_SHORT]
        sr_val = _cell_value(sr_coef, sr_p)
        sr_star = _sig_stars(sr_p)
        row_data.append(sr_val)
        row_annot.append(f"{sr_val:.3f}\n{sr_star}" if sr_star else f"{sr_val:.3f}" if sr_val != 0 else "")

        lr_p = fw_row["long_run_pvalue"]
        lr_coef = fw_row[COL_LONG]
        lr_val = _cell_value(lr_coef, lr_p)
        lr_star = _sig_stars(lr_p)
        row_data.append(lr_val)
        row_annot.append(f"{lr_val:.3f}\n{lr_star}" if lr_star else f"{lr_val:.3f}" if lr_val != 0 else "")
    else:
        row_data.extend([0.0, 0.0])
        row_annot.extend(["", ""])

    # Direction 2: climate predicting COVID-19 (reverse)
    rv_subset = df3[(df3["emotion"] == emotion) & (df3["direction"] == "reverse")]
    if not rv_subset.empty:
        rv_row = rv_subset.iloc[0]

        sr_p = rv_row["short_run_pvalue"]
        sr_coef = rv_row[COL_SHORT]
        sr_val = _cell_value(sr_coef, sr_p)
        sr_star = _sig_stars(sr_p)
        row_data.append(sr_val)
        row_annot.append(f"{sr_val:.3f}\n{sr_star}" if sr_star else f"{sr_val:.3f}" if sr_val != 0 else "")

        lr_p = rv_row["long_run_pvalue"]
        lr_coef = rv_row[COL_LONG]
        lr_val = _cell_value(lr_coef, lr_p)
        lr_star = _sig_stars(lr_p)
        row_data.append(lr_val)
        row_annot.append(f"{lr_val:.3f}\n{lr_star}" if lr_star else f"{lr_val:.3f}" if lr_val != 0 else "")
    else:
        row_data.extend([0.0, 0.0])
        row_annot.extend(["", ""])

    matrix_data.append(row_data)
    annot_data.append(row_annot)

# Define column names
columns = [
    "COVID-19 $\\rightarrow$ Climate\n(Short-run)",
    "COVID-19 $\\rightarrow$ Climate\n(Long-run)",
    "Climate $\\rightarrow$ COVID-19\n(Short-run)",
    "Climate $\\rightarrow$ COVID-19\n(Long-run)",
]

coef_df = pd.DataFrame(matrix_data, index=emotions_order, columns=columns)
annot_df = pd.DataFrame(annot_data, index=emotions_order, columns=columns)

# 3. Configure plotting
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
fig, ax = plt.subplots(figsize=(12, 8))

cmap = sns.color_palette("Reds", as_cmap=True)

mask = coef_df == 0
ax.set_facecolor("#F0F0F0")

vmax = float(np.nanmax(coef_df.values))
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

sns.heatmap(
    coef_df,
    annot=annot_df,
    fmt="",
    cmap=cmap,
    mask=mask,
    cbar_kws={"label": "Mutual Regression Coefficient ($\\beta$)", "shrink": 0.8},
    linewidths=2,
    linecolor="white",
    ax=ax,
    vmin=0,
    vmax=vmax,
)

ax.xaxis.tick_top()
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold", rotation=0)

ax.axvline(2, color="white", linewidth=6)
ax.axhline(4, color="white", linewidth=6)

plt.tight_layout()
plt.savefig(
    output_jpeg,
    format="jpeg",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={"quality": 95},
)
plt.close()
print(f"Data: {input_file}")
print(f"Export successful: {output_jpeg}")
