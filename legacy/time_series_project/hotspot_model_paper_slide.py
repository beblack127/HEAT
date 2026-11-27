"""Sparse black-and-white methods figure sized for one 16:9 slide."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
INK = "#111111"
MID = "#4A4A4A"
RULE = "#999999"


def arrow(ax, start, end, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=INK, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0",
    ))


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 14,
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=240)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13.333)
ax.set_ylim(0, 7.5)
ax.axis("off")

# Panel (a): statistical forecast.
ax.text(0.72, 6.72, "(a) Forecasting", fontsize=18, color=INK, va="center")
ax.plot([0.72, 12.62], [6.47, 6.47], color=RULE, lw=0.65)

items = [(5.65, "Mobility history"),
         (5.01, "Business licensing history"),
         (4.37, "Other observed covariates")]
for y, label in items:
    ax.text(1.04, y, label, fontsize=15, color=INK, va="center")
    ax.plot([4.45, 4.82], [y, y], color=INK, lw=0.9)
ax.plot([4.82, 4.82], [4.37, 5.65], color=INK, lw=0.9)
arrow(ax, (4.82, 5.01), (5.42, 5.01))

ax.add_patch(Rectangle((5.43, 4.59), 2.46, 0.84,
                       fill=False, edgecolor=INK, linewidth=1.0))
ax.text(6.66, 5.01, "ARIMAX / SARIMAX", fontsize=15,
        color=INK, ha="center", va="center")
arrow(ax, (7.90, 5.01), (8.62, 5.01))
ax.text(8.90, 5.25, "Mobility forecast", fontsize=15, color=INK,
        va="center")
ax.text(8.90, 4.68, r"$\hat{y}_{d,t+1:t+3}$", fontsize=20,
        color=INK, va="center")

# Panel (b): transparent assessment after forecasting.
ax.plot([0.72, 12.62], [3.67, 3.67], color=RULE, lw=0.65)
ax.text(0.72, 3.33, "(b) Hotspot assessment", fontsize=18,
        color=INK, va="center")

ax.text(1.04, 2.15, "Predicted mobility", fontsize=15,
        color=INK, va="center")
arrow(ax, (3.62, 2.15), (4.30, 2.15))
ax.add_patch(Rectangle((4.31, 1.67), 4.32, 0.96,
                       fill=False, edgecolor=INK, linewidth=1.0))
ax.text(6.47, 2.35, "Predefined criteria", fontsize=15,
        color=INK, ha="center", va="center")
ax.text(6.47, 1.91, "activity level   ·   growth   ·   persistence",
        fontsize=12.5, color=MID, ha="center", va="center")
arrow(ax, (8.64, 2.15), (9.32, 2.15))
ax.text(9.56, 2.15, "Candidate ranking", fontsize=15,
        color=INK, va="center")

ax.text(0.74, 0.62,
        "Only observed information available at the forecast date is used; criteria are set using development data.",
        fontsize=10.4, color=MID, va="center")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"hotspot_model_paper_slide.{ext}", dpi=240,
                bbox_inches=None, pad_inches=0, facecolor="white")
plt.close(fig)
