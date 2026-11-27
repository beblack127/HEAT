"""Make a presentation chart for the independent search crosscheck."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "weekly_hotplace_rank" / "external_validity"
data = pd.read_csv(SOURCE / "monthly_comparison.csv")
data = data.loc[data.fold.eq("test")].copy()
pivot = data.pivot(index="month", columns="candidate", values="search_yoy_median") * 100
months = [pd.Timestamp(m + "-01").strftime("%b") for m in pivot.index]
x = list(range(len(months)))

fig, ax = plt.subplots(figsize=(10, 4.9), dpi=180)
fig.patch.set_facecolor("white")
ax.axhline(0, color="#8994a3", linewidth=1)
ax.plot(x, pivot[True], marker="o", color="#ce563c", linewidth=2.4,
        label="Mobility-defined emerging candidates")
ax.plot(x, pivot[False], marker="o", color="#4269a5", linewidth=2.4,
        label="Other neighborhoods above volume floor")
ax.set_xticks(x, months)
ax.set_ylabel("Median YoY change in Seoul search share (%)")
ax.set_title("Independent search crosscheck of emerging-place proxy", loc="left", fontsize=14, pad=16)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.22)
ax.set_axisbelow(True)
fig.text(.08, .01, "Jan–Jul 2025; actual mobility candidates vs other high-volume dong-months.\n"
         "Target-month Naver search is used only for retrospective validation, not forecasting.",
         fontsize=8.5, color="#586575")
fig.subplots_adjust(bottom=.34, top=.84, left=.1, right=.96)
fig.savefig(SOURCE / "search_crosscheck_english.png", bbox_inches="tight")
