"""Slide-sized academic schematic of the proposed forecasting workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
INK = "#20252B"
SECONDARY = "#59636D"
RULE = "#C8CED3"
ACCENT = "#315C7D"
PALE = "#EAF0F4"


def line(ax, x1, y1, x2, y2, *, color=RULE, lw=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt")


def arrow(ax, x1, y1, x2, y2, *, color=ACCENT, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0",
    ))


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=240)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13.333)
ax.set_ylim(0, 7.5)
ax.axis("off")

# Title and page frame.
ax.text(0.68, 6.93, "Forecasting neighborhood activity and identifying hotspot candidates",
        fontsize=21.2, fontweight="semibold", color=INK, ha="left", va="center")
line(ax, 0.68, 6.54, 12.65, 6.54, color=ACCENT, lw=1.35)

# Three stage headings form the upper flow.
stages = [
    (0.74, "01", "Observed history"),
    (5.12, "02", "Forecasting model"),
    (9.22, "03", "Future activity"),
]
for x, number, label in stages:
    ax.text(x, 5.90, number, fontsize=10.5, color=ACCENT,
            fontweight="bold", ha="left", va="center")
    ax.text(x + 0.40, 5.90, label, fontsize=15.5, color=INK,
            fontweight="semibold", ha="left", va="center")
    line(ax, x, 5.64, x + (3.28 if number == "01" else 3.12), 5.64,
         color=RULE, lw=0.9)

# Input list: typographic structure instead of a card grid.
inputs = [
    (5.06, "Mobility history", "Past neighborhood arrival activity"),
    (4.23, "Business dynamics", "New licenses and closures, lagged"),
    (3.40, "Context indicators", "Observed subway or living-population measures"),
]
for y, heading, detail in inputs:
    line(ax, 0.75, y + 0.22, 0.75, y - 0.32, color=ACCENT, lw=1.8)
    ax.text(1.00, y + 0.12, heading, fontsize=14.2,
            fontweight="semibold", color=INK, va="center")
    ax.text(1.00, y - 0.21, detail, fontsize=10.9,
            color=SECONDARY, va="center")

# Flow into the model.
arrow(ax, 4.15, 4.25, 4.85, 4.25)
ax.add_patch(Rectangle((5.12, 3.70), 3.34, 1.12,
                       facecolor=PALE, edgecolor=ACCENT, linewidth=1.4))
ax.text(6.79, 4.36, "ARIMAX / SARIMAX", fontsize=17.6,
        fontweight="semibold", color=ACCENT, ha="center", va="center")
ax.text(6.79, 3.96, "monthly neighborhood series", fontsize=11.2,
        color=SECONDARY, ha="center", va="center")
ax.text(6.79, 3.38, "Reference models: naïve · seasonal naïve · ARIMA / SARIMA",
        fontsize=10.5, color=SECONDARY, ha="center", va="center")
arrow(ax, 8.72, 4.25, 9.10, 4.25)

# Forecast output, no implication that future exogenous observations are known.
ax.text(9.27, 4.73, "Predicted mobility indicator", fontsize=14.1,
        color=INK, fontweight="semibold", va="center")
ax.text(9.28, 4.14, r"$\hat{y}_{d,t+1:t+3}$", fontsize=27,
        color=ACCENT, va="center")
ax.text(9.28, 3.56, "Next three monthly observations", fontsize=11.1,
        color=SECONDARY, va="center")

# Distinct second stage: transparent, fixed assessment rule.
line(ax, 0.68, 2.90, 12.65, 2.90, color=RULE, lw=0.9)
ax.text(0.74, 2.51, "04", fontsize=10.5, color=ACCENT,
        fontweight="bold", va="center")
ax.text(1.14, 2.51, "Rule-based hotspot assessment", fontsize=15.5,
        color=INK, fontweight="semibold", va="center")

metrics = [
    (0.85, "Activity level", "How active is the area?"),
    (4.63, "Growth", "Change from the comparable period"),
    (8.23, "Persistence", "Is the change sustained?"),
]
for x, head, sub in metrics:
    ax.text(x, 1.82, head, fontsize=14.7, color=INK,
            fontweight="semibold", va="center")
    ax.text(x, 1.42, sub, fontsize=10.8, color=SECONDARY, va="center")
line(ax, 4.27, 1.26, 4.27, 2.06, color=RULE, lw=0.85)
line(ax, 7.88, 1.26, 7.88, 2.06, color=RULE, lw=0.85)

line(ax, 0.77, 0.97, 12.56, 0.97, color=RULE, lw=0.9)
ax.text(0.83, 0.63, "Output", fontsize=10.5, color=SECONDARY,
        fontweight="bold", va="center")
ax.text(1.58, 0.63, "Ranked hotspot candidates by administrative dong",
        fontsize=14.8, color=ACCENT, fontweight="semibold", va="center")
ax.text(12.55, 0.63, "→", fontsize=21, color=ACCENT,
        ha="right", va="center")

ax.text(0.78, 0.16,
        "Only information available at the forecast date enters the model; assessment criteria are fixed using development data.",
        fontsize=8.3, color=SECONDARY, va="center")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"hotspot_model_fullslide.{ext}", dpi=240,
                bbox_inches=None, pad_inches=0, facecolor="white")
plt.close(fig)
