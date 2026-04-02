import matplotlib

matplotlib.use("Agg")  # 无界面后端，不弹出窗口，仅导出文件
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from pathlib import Path

# =========================
# 1. Figure setup
# =========================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis('off')

# =========================
# 2. Color palette
# =========================
c_covid = '#D55E00'      # vermilion
c_climate = '#009E73'    # bluish green
c_both = '#0072B2'       # deep blue
c_text_dark = '#212529'
c_text_muted = '#6C757D'
c_light_edge = '#ADB5BD'
c_light_fill = '#F8F9FA'
c_theme_fill = '#F4F7FB'
c_arrow = '#CED4DA'

# =========================
# 3. Helper functions
# =========================
def draw_node(x, y, w, h, color, text1, text2):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.5",
        linewidth=2,
        edgecolor=color,
        facecolor='white',
        zorder=4
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2 + 0.3, text1,
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        color=color, zorder=5
    )
    ax.text(
        x + w / 2, y + h / 2 - 0.4, text2,
        ha='center', va='center',
        fontsize=13, fontweight='bold',
        color=c_text_muted, zorder=5
    )

def draw_theme_box(x, y, w, h, title, emos, quotes):
    # Background box
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        linewidth=1.5,
        edgecolor=c_both,
        facecolor=c_theme_fill,
        alpha=0.95,
        zorder=2
    )
    ax.add_patch(box)

    # Theme title（多行高度用行数估算，避免与情绪条叠字）
    title_top_y = y + h - 0.38
    ax.text(
        x + w / 2, title_top_y, title,
        ha='center', va='top',
        fontsize=13.5, fontweight='bold',
        color=c_text_dark, zorder=5,
        linespacing=1.25,
    )
    n_title_lines = title.count("\n") + 1
    # 略小于真实行高时可缩小「标题最后一行」与情绪条之间的空白（0.46 会偏松）
    line_h = 0.38
    title_bottom = title_top_y - n_title_lines * line_h

    # Mapped emotions（贴在标题下方，再留空隙给引文）
    gap_title_to_badge = 0.0
    badge_h = 0.42
    badge_ll_y = title_bottom - gap_title_to_badge - badge_h
    badge = patches.FancyBboxPatch(
        (x + w / 2 - 2.45, badge_ll_y), 4.9, badge_h,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        linewidth=0,
        facecolor=c_both,
        alpha=0.15,
        zorder=3,
    )
    ax.add_patch(badge)
    ax.text(
        x + w / 2, badge_ll_y + badge_h / 2,
        f"{emos}",
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        color=c_both, zorder=5,
    )

    gap_badge_to_quotes = 0.28
    curr_y = badge_ll_y - gap_badge_to_quotes

    # Quotes
    for q in quotes:
        wrapped_q = "\n".join(textwrap.wrap(f'"{q}"', width=42))
        ax.text(
            x + 0.3, curr_y, wrapped_q,
            ha='left', va='top',
            fontsize=10.8, fontstyle='italic',
            color=c_text_dark, zorder=5
        )
        lines = len(wrapped_q.split('\n'))
        quote_line_step = 0.32
        quote_block_gap = 0.26
        curr_y -= lines * quote_line_step + quote_block_gap

# =========================
# 4. Top nodes
# =========================
draw_node(3, 12.5, 5, 1.5, c_covid, "COVID-19 Pandemic", "(Pandemic severity)")
draw_node(12, 12.5, 5, 1.5, c_climate, "Climate Crisis", "(Existential threat)")

# Psychological distance
ax.plot([8.5, 11.5], [13.25, 13.25],
        color='#DEE2E6', linewidth=3, linestyle='--', zorder=1)
ax.text(
    10, 13.5, "Psychological Distance",
    ha='center', va='center',
    fontsize=11, fontstyle='italic',
    color=c_text_muted
)

# =========================
# 5. Main arrows to central qualitative bridge
# =========================
kw_main = dict(
    arrowstyle="Simple, tail_width=2.5, head_width=8, head_length=8",
    color=c_arrow, zorder=1
)

ax.add_patch(
    patches.FancyArrowPatch(
        (5.5, 12.4), (9.5, 11.0),
        connectionstyle="arc3,rad=0.1",
        **kw_main
    )
)
ax.add_patch(
    patches.FancyArrowPatch(
        (14.5, 12.4), (10.5, 11.0),
        connectionstyle="arc3,rad=-0.1",
        **kw_main
    )
)

# =========================
# 6. Central title
# =========================
ax.text(
    10, 10.5,
    "Illustrative Themes in High-Intensity Dual-Crisis Discourse",
    ha='center', va='center',
    fontsize=16, fontweight='bold',
    color=c_both, zorder=5
)

# =========================
# 7. Theme content (updated to latest manuscript/supplementary)
# =========================
t1_qs = [
    "Lost my teenage years to pressure to get into college and sick parents, lost my 20s to recession plus chronic illness developed after catching swine flu, losing my 30s to pandemic, lost dating to pandemic and illness, losing rest of my life to climate crisis. COOL COOL",
    "You think coronavirus is bad, wait till you meet climate change...."
]

t2_qs = [
    "Republicans cannot help being willfully ignorant. Deny COVID-19, deny vaccines, deny climate change, deny evolution, deny science, deny reality.",
    "Science doesn't require belief but making good policy choices on things like covid or global warming does require a certain level of trust in scientific expertise."
]

t3_qs = [
    "The experience of the coronavirus pandemic actually provides some semblance of hope that we can get our shit together to deal with the climate emergency.",
    "Exactly. Look how quickly we got a safe and working vaccine for covid when we threw money at that problem. Imagine what could happen if the government threw that kind of money, resources, and urgency at climate change."
]

draw_theme_box(
    1.5, 4.0, 5.2, 5.8,
    "Theme 1\nCrises that build on each other\nand their relative severity",
    "Sadness, Fear, Disgust",
    t1_qs,
)

draw_theme_box(
    7.4, 4.0, 5.2, 5.8,
    "Theme 2\nThe crisis of collective action\nand scientific authority",
    "Anger, Trust",
    t2_qs,
)

draw_theme_box(
    13.3, 4.0, 5.5, 5.8,
    "Theme 3\nSilver linings and\nfuture expectations",
    "Surprise, Anticipation, Joy",
    t3_qs,
)

# =========================
# 8. Bottom summary arrows
# =========================
kw_bottom = dict(
    arrowstyle="Simple, tail_width=1.5, head_width=6, head_length=6",
    color=c_arrow, zorder=1
)

ax.add_patch(patches.FancyArrowPatch((4.1, 3.8), (8, 2.3), **kw_bottom))
ax.add_patch(patches.FancyArrowPatch((10, 3.8), (10, 2.3), **kw_bottom))
ax.add_patch(patches.FancyArrowPatch((15.9, 3.8), (12, 2.3), **kw_bottom))

# =========================
# 9. Bottom synthesis box (updated to more neutral wording)
# =========================
box_bot = patches.FancyBboxPatch(
    (2, 0.2), 16, 1.8,
    boxstyle="round,pad=0.2,rounding_size=0.4",
    linewidth=2,
    edgecolor=c_light_edge,
    facecolor=c_light_fill,
    zorder=4
)
ax.add_patch(box_bot)

ax.text(
    10, 1.4,
    "INTERPRETIVE LINKAGES IN DUAL-CRISIS DISCOURSE",
    ha='center', va='center',
    fontsize=16, fontweight='bold',
    color=c_text_dark, zorder=5
)

ax.text(
    10, 0.7,
    "Cascading risk   +   collective action and scientific authority   +   future-oriented expectations",
    ha='center', va='center',
    fontsize=12.8, fontweight='bold',
    color=c_text_muted, zorder=5
)

# =========================
# 10. Export
# =========================
_output_dir = Path(r"F:\Figure\Fig5")
_output_dir.mkdir(parents=True, exist_ok=True)
output_path = _output_dir / "Figure5_final.jpeg"
plt.savefig(
    output_path,
    format='jpeg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    pil_kwargs={'quality': 95}
)

print(f"Export completed! Saved as: {output_path}")
plt.close(fig)