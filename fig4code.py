import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

# 1. Initialize figure and font settings
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis('off')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']

# 2. Define global color palette (Okabe-Ito Palette)
c_covid = '#D55E00'   # Refined vermilion
c_climate = '#009E73' # Refined bluish green
c_both = '#0072B2'    # Deep ocean blue
c_text_dark = '#212529'
c_text_muted = '#6C757D'

# 3. Draw top nodes (two crises)
def draw_node(x, y, w, h, color, text1, text2):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.5", 
                                 linewidth=2, edgecolor=color, facecolor='white', zorder=4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.3, text1, ha='center', va='center', fontsize=18, fontweight='bold', color=color, zorder=5)
    ax.text(x + w/2, y + h/2 - 0.4, text2, ha='center', va='center', fontsize=13, fontweight='bold', color=c_text_muted, zorder=5)

draw_node(3, 12.5, 5, 1.5, c_covid, "COVID-19 Pandemic", "(Mortality Salience)")
draw_node(12, 12.5, 5, 1.5, c_climate, "Climate Crisis", "(Existential Threat)")

# Draw dashed line representing initial "psychological distance"
ax.plot([8.5, 11.5], [13.25, 13.25], color='#DEE2E6', linewidth=3, linestyle='--', zorder=1)
ax.text(10, 13.5, "Psychological Distance", ha='center', va='center', fontsize=11, fontstyle='italic', color=c_text_muted)

# ==========================================
# Core: each crisis sends one main arrow toward central bridging
# ==========================================
# Use thick main arrows with gentle curvature
kw_main = dict(arrowstyle="Simple, tail_width=2.5, head_width=8, head_length=8", color='#CED4DA', zorder=1)

# COVID-19 (left) -> central convergence
ax.add_patch(patches.FancyArrowPatch((5.5, 12.4), (9.5, 11.0), connectionstyle="arc3,rad=0.1", **kw_main))
# Climate (right) -> central convergence
ax.add_patch(patches.FancyArrowPatch((14.5, 12.4), (10.5, 11.0), connectionstyle="arc3,rad=-0.1", **kw_main))

# 4. Draw central core (three qualitative themes)
ax.text(10, 10.5, "Cognitive Thematic Bridging (Dual-Crisis Discourse)", ha='center', va='center', fontsize=16, fontweight='bold', color=c_both, zorder=5)

def draw_theme_box(x, y, w, h, title, emos, quotes):
    # Background box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.3", 
                                 linewidth=1.5, edgecolor=c_both, facecolor='#F4F7FB', alpha=0.9, zorder=2)
    ax.add_patch(box)
    # Title
    ax.text(x + w/2, y + h - 0.5, title, ha='center', va='top', fontsize=14, fontweight='bold', color=c_text_dark, zorder=5)
    # Emotion label band (shifted downward and slightly widened)
    badge = patches.FancyBboxPatch((x + w/2 - 2.4, y + h - 1.7), 4.8, 0.4, boxstyle="round,pad=0.1,rounding_size=0.2", 
                                 linewidth=0, facecolor=c_both, alpha=0.15, zorder=3)
    ax.add_patch(badge)
    ax.text(x + w/2, y + h - 1.45, f"Mapped Emotions: {emos}", ha='center', va='center', fontsize=10, fontweight='bold', color=c_both, zorder=5)
    
    # Auto-wrap and render illustrative Reddit quotes
    curr_y = y + h - 2.2
    for q in quotes:
        wrapped_q = "\n".join(textwrap.wrap(f'"{q}"', width=42))
        ax.text(x + 0.3, curr_y, wrapped_q, ha='left', va='top', fontsize=11, fontstyle='italic', color=c_text_dark, zorder=5)
        lines = len(wrapped_q.split('\n'))
        curr_y -= (lines * 0.35 + 0.5)

# --- Selected quotes for each theme (from CSV analysis) ---
t1_qs = [
    "Lost my teenage years to pressure to get into college and sick parents, lost my 20s to recession plus chronic illness developed after catching swine flu, losing my 30s to pandemic, lost dating to pandemic and illness, losing rest of my life to climate crisis. COOL COOL",
    "You think coronavirus is bad, wait till you meet climate change...."
]
t2_qs = [
    "If we can't come together to fight against a disease how will we come together to fight climate change.",
    "Making good policy choices on things like covid or global warming does require a certain level of trust in scientific expertise."
]
t3_qs = [
    "The experience of the coronavirus pandemic actually provides some semblance of hope that we can get our shit together...",
    "I really hope we start tackling climate change and environmental issues. It will prevent a lot of disasters- including another pandemic. Hopefully the green recovery plans will be a good start",
]
#"Right now a vaccine for COVID-19 would be pretty sweet. On a longer timeframe, things to help deal with Climate Change... would be pleasant."
draw_theme_box(1.5, 4.0, 5.2, 5.8, "Theme 1\nCascading Crises", "Sadness, Fear, Disgust", t1_qs)
draw_theme_box(7.4, 4.0, 5.2, 5.8, "Theme 2\nCollective Action Failure", "Anger, Trust", t2_qs)
draw_theme_box(13.3, 4.0, 5.5, 5.8, "Theme 3\nAnticipated Solutions", "Surprise, Anticipation, Joy", t3_qs)

# 5. Draw bottom summary node (theoretical synthesis)
kw_bottom = dict(arrowstyle="Simple, tail_width=1.5, head_width=6, head_length=6", color='#CED4DA', zorder=1)
# Downward converging arrows (from theme boxes to slightly above the bottom node to avoid pointing inside)
ax.add_patch(patches.FancyArrowPatch((4.1, 3.8), (8, 2.3), **kw_bottom))
ax.add_patch(patches.FancyArrowPatch((10, 3.8), (10, 2.3), **kw_bottom))
ax.add_patch(patches.FancyArrowPatch((15.9, 3.8), (12, 2.3), **kw_bottom))

# Dark high-contrast bottom base
box_bot = patches.FancyBboxPatch((2, 0.2), 16, 1.8, boxstyle="round,pad=0.2,rounding_size=0.4", 
                                 linewidth=2, edgecolor='#111111', facecolor='#212529', zorder=4)
ax.add_patch(box_bot)

ax.text(10, 1.4, "AFFECT GENERALIZATION (Dynamic Psychological Resonance)", ha='center', va='center', fontsize=16, fontweight='bold', color='white', zorder=5)
ax.text(10, 0.7, "Deep-Seated Trauma (Sadness, Disgust)   +   Resilient Scientific Reliance (Trust, Surprise)", ha='center', va='center', fontsize=14, fontweight='bold', color='#FFD166', zorder=5)

# 6. Save output (PNG only)
plt.savefig(r'E:\Figure\Fig4\Figure4.png', format='png', dpi=300, bbox_inches='tight')
print("Export completed! Minimal funnel-structure diagram generated (PNG).")