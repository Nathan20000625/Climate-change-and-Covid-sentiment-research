import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# 1. Load real data (using specified absolute path)
data_dir = r"E:\Figure\Fig1"
df_a = pd.read_csv(os.path.join(data_dir, "fig1a-data.csv"))
df_b = pd.read_csv(os.path.join(data_dir, "fig1b-data.csv"))
df_c = pd.read_csv(os.path.join(data_dir, "fig1c-data.csv"))

# 2. Preprocess Panel A data (compute 7-day moving average)
df_a['date'] = pd.to_datetime(df_a['date'])
df_a = df_a.sort_values('date')
df_a['Climate_MA'] = df_a['Climate'].rolling(window=7, min_periods=1).mean()
df_a['Death_MA'] = df_a['US_daily_covid_death'].rolling(window=7, min_periods=1).mean()

# 3. Set global plotting style
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif'] # Use sans-serif fonts
plt.rcParams['axes.spines.top'] = False    # Remove top spine
plt.rcParams['axes.spines.right'] = False  # Remove right spine
plt.rcParams['axes.linewidth'] = 1.0       # Spine linewidth

# ---------------------------------------------------------
# --- Define Okabe-Ito color palette ---
# ---------------------------------------------------------
color_climate = '#009E73'   # Bluish Green, for climate-related series
color_covid = '#D55E00'     # Vermilion, for COVID-related series
color_neutral = '#333333'   # Dark slate neutral gray (mainly for Panel b)
color_baseline = '#888888'  # Soft gray (for baseline dashed line)


def plot_panel_a(ax):
    # Plot climate comments line (left y-axis)
    line1, = ax.plot(
        df_a['date'],
        df_a['Climate_MA'],
        color=color_climate,
        linewidth=2,
        label='Climate Change Comments (7-day MA)'
    )
    ax.set_ylabel('Climate Change Comments Volume', color=color_climate, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_climate)

    # Plot COVID-19 deaths area (right y-axis)
    ax_twin = ax.twinx()
    ax_twin.spines['top'].set_visible(False)
    ax_twin.spines['left'].set_visible(False)
    fill1 = ax_twin.fill_between(
        df_a['date'],
        df_a['Death_MA'],
        color=color_covid,
        alpha=0.3,
        label='COVID-19 Deaths (7-day MA)'
    )
    ax_twin.set_ylabel('COVID-19 Daily Deaths', color=color_covid, fontsize=12, fontweight='bold')
    ax_twin.tick_params(axis='y', labelcolor=color_covid)

    # Format date axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title('a', loc='left', fontweight='bold', fontsize=16, x=-0.05)

    # Combine legends
    lines = [line1, fill1]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', frameon=False, fontsize=11)


def plot_panel_b(ax):
    # Compute upper and lower errors for 95% CI
    df_b_local = df_b.copy()
    df_b_local['err_low'] = df_b_local['coefficient'] - df_b_local['CI95_low']
    df_b_local['err_high'] = df_b_local['CI95_high'] - df_b_local['coefficient']
    # Reverse order so plot matches table from top to bottom
    df_b_local = df_b_local.iloc[::-1].reset_index(drop=True)

    # Customize tighter y-axis positions
    if len(df_b_local) == 2:
        y_pos_b = np.array([0.3, 0.7])
        ax.set_ylim(0.2, 0.8)
    else:
        y_pos_b = np.linspace(0, 1, len(df_b_local))

    ax.errorbar(
        df_b_local['coefficient'],
        y_pos_b,
        xerr=[df_b_local['err_low'], df_b_local['err_high']],
        fmt='o',
        color=color_neutral,
        ecolor=color_neutral,
        capsize=5,
        elinewidth=1.5,
        markersize=8
    )
    ax.axvline(0, color=color_baseline, linestyle='--', linewidth=1)  # Add x=0 reference line
    ax.set_yticks(y_pos_b)

    labels_b = df_b_local['independent_variable'].replace({
        'US_daily_covid_confirm': 'Confirmed Cases',
        'US_daily_covid_death': 'COVID-19 Deaths'
    })
    ax.set_yticklabels(labels_b, fontsize=11)
    ax.set_xlabel('Standardized OLS Coefficient (log-log)', fontsize=11)
    ax.set_title('b', loc='left', fontweight='bold', fontsize=16, x=-0.1)

    # Automatically add significance stars
    for i, row in df_b_local.iterrows():
        pval = row['pval']
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        if sig:
            ax.text(
                row['CI95_high'] + 0.002,
                y_pos_b[i],
                sig,
                va='center',
                ha='left',
                fontsize=12,
                fontweight='bold',
                color=color_neutral
            )


def plot_panel_c(ax):
    df_c_local = df_c.copy()
    df_c_local['err_low'] = df_c_local['coefficient'] - df_c_local['CI95_low']
    df_c_local['err_high'] = df_c_local['CI95_high'] - df_c_local['coefficient']
    df_c_local = df_c_local.iloc[::-1].reset_index(drop=True)

    y_pos_c = np.arange(len(df_c_local))
    colors_c = [color_covid if x < 0 else color_climate for x in df_c_local['coefficient']]

    for i, row in df_c_local.iterrows():
        ax.errorbar(
            row['coefficient'],
            i,
            xerr=[[row['err_low']], [row['err_high']]],
            fmt='o',
            color=colors_c[i],
            ecolor=colors_c[i],
            capsize=5,
            elinewidth=1.5,
            markersize=8
        )

    ax.axvline(0, color=color_baseline, linestyle='--', linewidth=1)
    ax.set_yticks(y_pos_c)
    ax.set_yticklabels(df_c_local['dependent_variable'].str.capitalize(), fontsize=11)
    ax.set_xlabel('Standardized OLS Coefficient', fontsize=11)
    ax.set_title('c', loc='left', fontweight='bold', fontsize=16, x=-0.1)

    # Automatically add significance stars
    for i, row in df_c_local.iterrows():
        pval = row['pval']
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        if sig:
            ax.text(
                row['CI95_high'] + 0.01,
                i,
                sig,
                va='center',
                ha='left',
                color=colors_c[i],
                fontsize=12,
                fontweight='bold'
            )


# Initialize figure (aspect ratio and 1+2 grid layout)
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2])
ax1 = fig.add_subplot(gs[0, :])   # Panel a occupies the entire first row
ax2 = fig.add_subplot(gs[1, 0])   # Panel b occupies the left of the second row
ax3 = fig.add_subplot(gs[1, 1])   # Panel c occupies the right of the second row

# Draw three panels on the main figure
plot_panel_a(ax1)
plot_panel_b(ax2)
plot_panel_c(ax3)

# 4. Adjust layout and export high-resolution images (JPEG, 300 dpi)
plt.tight_layout(pad=3.0)
output_dir = data_dir
os.makedirs(output_dir, exist_ok=True)

_jpeg_kw = dict(format='jpeg', dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})

# 4.1 Export full Figure 1
output_jpeg_path = os.path.join(output_dir, "Figure_1.jpeg")
plt.savefig(output_jpeg_path, **_jpeg_kw)

# 4.2 Separately redraw and export three standalone subplots (with full axes)
# Panel a (slightly taller)
fig_a, ax_a = plt.subplots(figsize=(10, 4.5))
plot_panel_a(ax_a)
fig_a.tight_layout(pad=2.0)
fig_a.savefig(os.path.join(output_dir, "Figure_1a.jpeg"), **_jpeg_kw)
plt.close(fig_a)

# Panel b
fig_b, ax_b = plt.subplots(figsize=(6, 4))
plot_panel_b(ax_b)
fig_b.tight_layout(pad=2.0)
fig_b.savefig(os.path.join(output_dir, "Figure_1b.jpeg"), **_jpeg_kw)
plt.close(fig_b)

# Panel c
fig_c, ax_c = plt.subplots(figsize=(8, 4))
plot_panel_c(ax_c)
fig_c.tight_layout(pad=2.0)
fig_c.savefig(os.path.join(output_dir, "Figure_1c.jpeg"), **_jpeg_kw)
plt.close(fig_c)

print(f"Export successful! JPEG images saved to: {output_jpeg_path}, and 3 separate subplot JPEG files.")