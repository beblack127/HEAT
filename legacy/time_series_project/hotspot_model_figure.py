"""Minimal, journal-style schematic of the proposed forecasting workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
BLACK = "#171717"
GRAY = "#585858"
RULE = "#A5A5A5"


def arrow(ax, x1, y1, x2, y2, lw=0.8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=BLACK, shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0",
    ))


def rectangle(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                           edgecolor=BLACK, linewidth=0.8))


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.3, 3.05), dpi=400)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 7.3)
ax.set_ylim(0, 3.05)
ax.axis("off")

# (a) Forecasting: observed histories enter one statistical model.
ax.text(0.15, 2.83, "a", fontsize=10, fontweight="bold", va="center")
ax.text(0.35, 2.83, "Forecasting", fontsize=9.3, va="center")
ax.plot([0.15, 7.13], [2.67, 2.67], color=RULE, lw=0.55)

ax.text(0.35, 2.27, "Mobility history", fontsize=8.6, va="center")
ax.text(0.35, 1.96, "Business licensing history", fontsize=8.6, va="center")
ax.text(0.35, 1.65, "Other observed covariates", fontsize=8.6, va="center")
ax.plot([2.45, 2.45], [1.65, 2.27], color=BLACK, lw=0.75)
for y in (1.65, 1.96, 2.27):
    ax.plot([2.19, 2.45], [y, y], color=BLACK, lw=0.75)
arrow(ax, 2.45, 1.96, 2.86, 1.96)

rectangle(ax, 2.87, 1.70, 1.35, 0.52)
ax.text(3.545, 1.96, "ARIMAX / SARIMAX", fontsize=8.2,
        ha="center", va="center")
arrow(ax, 4.23, 1.96, 4.64, 1.96)
ax.text(4.76, 2.10, "Mobility forecasts", fontsize=8.8, va="center")
ax.text(4.76, 1.78, r"$\hat{y}_{d,t+1:t+3}$", fontsize=9.2, va="center")

# (b) Assessment is a separate fixed rule applied to the forecast.
ax.plot([0.15, 7.13], [1.38, 1.38], color=RULE, lw=0.55)
ax.text(0.15, 1.19, "b", fontsize=10, fontweight="bold", va="center")
ax.text(0.35, 1.19, "Assessment", fontsize=9.3, va="center")

ax.text(0.35, 0.71, "Predicted mobility", fontsize=8.6, va="center")
arrow(ax, 1.92, 0.71, 2.32, 0.71)
rectangle(ax, 2.33, 0.42, 2.22, 0.57)
ax.text(3.44, 0.78, "Predefined criteria", fontsize=8.5,
        ha="center", va="center")
ax.text(3.44, 0.57, "level  ·  growth  ·  persistence", fontsize=7.9,
        ha="center", va="center", color=GRAY)
arrow(ax, 4.56, 0.71, 4.96, 0.71)
ax.text(5.06, 0.71, "Hotspot candidate ranking", fontsize=8.6, va="center")

ax.text(0.35, 0.15,
        "Criteria are specified using development data before retrospective evaluation.",
        fontsize=7.3, color=GRAY, va="center")

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"hotspot_model_figure.{ext}", dpi=400,
                bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.close(fig)
