import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Input and output paths
INPUT_FILE = r'E:\Figure\Fig2\fig2-data.csv'
OUTPUT_DIR = r'E:\Figure\Fig2'

# 1. Load and transform data
df = pd.read_csv(INPUT_FILE)

# Convert data from wide to long format (for seaborn violin plot)
df_melt = df.drop(columns=['date']).melt(var_name='col_name', value_name='score')

# Extract group and emotion labels
df_melt['group'] = df_melt['col_name'].apply(lambda x: x.split('_')[0].capitalize())
df_melt['emotion'] = df_melt['col_name'].apply(lambda x: x.split('_')[1].capitalize())

# Define panel order
group_order = ['Climate', 'Covid', 'Both']
panel_a_emotions = ['Anger', 'Sadness', 'Fear', 'Disgust']
panel_b_emotions = ['Joy', 'Anticipation', 'Trust', 'Surprise']

# 2. Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.0

# --- Define color palette (Okabe-Ito + deep blue) ---
color_climate = '#009E73' # Refined bluish green
color_covid = '#D55E00'   # Refined vermilion
color_both = '#0072B2'    # Deep ocean blue (professional, highlights Both group)
palette = {'Climate': color_climate, 'Covid': color_covid, 'Both': color_both}

# 3. Initialize figure (2 rows × 1 column)
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# ---------------------------------------------------------
# --- Panel A (panel a): high-arousal negative emotions ---
# ---------------------------------------------------------
df_a = df_melt[df_melt['emotion'].isin(panel_a_emotions)]
sns.violinplot(data=df_a, x='emotion', y='score', hue='group', hue_order=group_order, 
               order=panel_a_emotions, palette=palette, inner='quartile', 
               ax=axes[0], cut=0, density_norm='width', linewidth=1.2)

# Panel A: title and axis labels
axes[0].set_title('a', loc='left', fontweight='bold', fontsize=18, x=-0.05, y=1.05)
axes[0].set_ylabel('Emotion Intensity Score', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', labelsize=12)
axes[0].legend(title='', frameon=False, fontsize=11, loc='upper right')

# Add ANOVA *** significance markers for Panel A
for i, emotion in enumerate(panel_a_emotions):
    y_max = df_a[df_a['emotion'] == emotion]['score'].max()
    y_star = y_max + 0.005 # Slightly above maximum value
    axes[0].text(i, y_star, '***', ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0].plot([i-0.25, i+0.25], [y_star-0.001, y_star-0.001], color='k', lw=1)

# ---------------------------------------------------------
# --- Panel B (panel b): positive and cognitive emotions ---
# ---------------------------------------------------------
df_b = df_melt[df_melt['emotion'].isin(panel_b_emotions)]
sns.violinplot(data=df_b, x='emotion', y='score', hue='group', hue_order=group_order, 
               order=panel_b_emotions, palette=palette, inner='quartile', 
               ax=axes[1], cut=0, density_norm='width', linewidth=1.2)

# Panel B: title and axis labels
axes[1].set_title('b', loc='left', fontweight='bold', fontsize=18, x=-0.05, y=1.05)
axes[1].set_ylabel('Emotion Intensity Score', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', labelsize=12)
axes[1].get_legend().remove() # Remove duplicated legend at bottom

# Add ANOVA *** significance markers for Panel B
for i, emotion in enumerate(panel_b_emotions):
    y_max = df_b[df_b['emotion'] == emotion]['score'].max()
    y_star = y_max + 0.005
    axes[1].text(i, y_star, '***', ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[1].plot([i-0.25, i+0.25], [y_star-0.001, y_star-0.001], color='k', lw=1)

# 4. Slightly increase y-axis upper limit to avoid cutting off stars
axes[0].set_ylim(bottom=0, top=axes[0].get_ylim()[1] + 0.01)
axes[1].set_ylim(bottom=0, top=axes[1].get_ylim()[1] + 0.01)

# 5. Export and save to specified output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.tight_layout(pad=3.0)
plt.savefig(
    os.path.join(OUTPUT_DIR, 'Figure_2.jpeg'),
    format='jpeg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    pil_kwargs={'quality': 95},
)

print(f"Export successful! Figure saved to {OUTPUT_DIR}: Figure_2.jpeg")