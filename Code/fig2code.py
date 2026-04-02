import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# 1. 读取并预处理数据
df = pd.read_csv(r'F:\Figure\Fig2\fig2-data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 兼容比例列名：优先 Both_ratio，其次 Covid_ratio
ratio_col = 'Both_ratio' if 'Both_ratio' in df.columns else 'Covid_ratio'
if ratio_col not in df.columns:
    raise KeyError(f"未找到比例列，当前可用列：{list(df.columns)}")

# 清洗比例列（去掉百分号并转换为浮点型小数）
df['Both_ratio'] = df[ratio_col].astype(str).str.rstrip('%').astype('float') / 100.0

# 2. 计算 7 天滑动平均 (7-day MA)
df['Climate_MA'] = df['Climate'].rolling(window=7, min_periods=1).mean()
df['Both_ratio_MA'] = df['Both_ratio'].rolling(window=7, min_periods=1).mean()

# （可选）计算皮尔森相关系数，仅用于验证，不在图中显示
corr_df = df[['Climate', 'Both_ratio']].dropna()
r, p_val = pearsonr(corr_df['Climate'], corr_df['Both_ratio'])
print(f"Pearson's r = {r:.3f}, p-value = {p_val:.3e}")

# 3. 开始绘图 (Nature Communications 标准样式)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.spines.top'] = False # 隐藏顶部边框
plt.rcParams['axes.linewidth'] = 1.0    # 坐标轴线宽

fig, ax1 = plt.subplots(figsize=(10, 6))

# 定义高级色盲友好配色 (Okabe-Ito Palette)
color_climate = '#009E73' # 蓝绿色 (Climate Volume)
color_ratio = '#D55E00'   # 朱红色 (Both Ratio)

# ----------------- 主坐标轴 (左侧：Climate Volume) -----------------
ax1.plot(df['Date'], df['Climate_MA'], color=color_climate, linewidth=2.5, label='Climate Volume (7-day MA)')

# 设置黑色轴线、标签和刻度
ax1.set_xlabel('')
ax1.set_ylabel('Climate Change Comments Volume', color='black', fontsize=13, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black', labelsize=11, color='black')
ax1.tick_params(axis='x', labelcolor='black', labelsize=11, color='black')
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')

# 格式化 X 轴时间标签 (每三个月显示一次)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# ----------------- 副坐标轴 (右侧：Both Ratio) -----------------
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_color('black') # 右侧轴线设为黑色

ax2.plot(df['Date'], df['Both_ratio_MA'], color=color_ratio, linewidth=2.5, alpha=0.85, label='Co-mention Ratio (7-day MA)')

# 设置右侧黑色标签和刻度
ax2.set_ylabel('Proportion of Co-mentions (Both Ratio)', color='black', fontsize=13, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='black', labelsize=11, color='black')

# 将右侧 Y 轴格式化为百分比形式 (如 5.0%, 10.0%)
from matplotlib.ticker import PercentFormatter
ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))

# ----------------- 图例设置 (Legend) -----------------
# 收集两条线的图例，并放置在图表底部外侧，防止遮挡数据
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
           ncol=2, frameon=False, fontsize=12)

# 自动紧凑布局
plt.tight_layout()

# 仅保存一张 300 DPI 的 JPEG 图片（不弹窗）
plt.savefig(r'F:\Figure\Fig2\Figure_2.jpeg', format='jpeg', dpi=300, bbox_inches='tight')
plt.close(fig)