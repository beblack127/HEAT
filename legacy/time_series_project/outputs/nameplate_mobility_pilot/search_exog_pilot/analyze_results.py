"""Summarize paired errors and draw an English midterm chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
PRED = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})

wide = PRED.pivot(index=["admin_dong_code", "month"], columns="model", values="abs_error")
assert wide.notna().all().all()
dong = PRED.groupby(["admin_dong_code", "model"], as_index=False).agg(
    abs_error=("abs_error", "sum"), actual=("actual_daily", "sum")
)
dong["wape_pct"] = 100 * dong.abs_error / dong.actual
median_wape = dong.groupby("model").wape_pct.median().round(2).to_dict()
win_counts = {m: int((wide[m] < wide["ARIMA"]).sum()) for m in wide if m != "ARIMA"}
win_dongs = {}
for m in wide:
    if m == "ARIMA":
        continue
    grouped = wide.groupby(level="admin_dong_code").sum()
    win_dongs[m] = int((grouped[m] < grouped["ARIMA"]).sum())
largest_losses = (wide["ARIMAX_search"] - wide["ARIMA"]).groupby(level="admin_dong_code").sum().sort_values(ascending=False).head(8)
names = pd.read_csv(OUT.parent / "nameplate_admin_dong_427.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
name_map = names.set_index("admin_dong_code").apply(lambda r: f"{r.gu} {r.dong_name}", axis=1).to_dict()
without_largest = PRED.loc[PRED.admin_dong_code.ne(largest_losses.index[0])]
without_largest_wape = (100 * without_largest.groupby("model").abs_error.sum() /
                        without_largest.groupby("model").actual_daily.sum()).round(2).to_dict()
diagnostics = {
    "median_dong_wape_pct": median_wape,
    "rows_better_than_ARIMA_out_of_320": win_counts,
    "dongs_better_than_ARIMA_out_of_40": win_dongs,
    "largest_search_vs_arima_loss_dongs": {f"{code} {name_map.get(code, '')}": round(float(loss), 1)
                                            for code, loss in largest_losses.items()},
    "wape_pct_excluding_largest_search_loss_dong_posthoc": without_largest_wape,
}
(OUT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
models = ["ARIMA", "ARIMAX_living", "ARIMAX_search", "ARIMAX_both"]
labels = ["Mobility only\nARIMA", "+ Living population\nARIMAX", "+ Naver search\nARIMAX", "+ Both inputs\nARIMAX"]
values = [summary["models"][m]["wape_pct"] for m in models]
colors = ["#00868b", "#b3bdc5", "#e59b40", "#c86f37"]
fig, ax = plt.subplots(figsize=(11.5, 5.3), facecolor="white")
bars = ax.bar(np.arange(4), values, color=colors, width=0.56)
ax.set_ylim(0, 17)
ax.set_ylabel("WAPE (%)  ·  lower is better", fontsize=13, color="#314b5e")
ax.set_xticks(np.arange(4), labels, fontsize=11)
ax.set_yticks([0, 5, 10, 15])
ax.grid(axis="y", alpha=.15)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, value + .25, f"{value:.2f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color="#163047")
ax.set_title("Naver search did not improve this mobility pilot", loc="left", fontsize=19, fontweight="bold", color="#163047", pad=25)
fig.text(.12, .025, "40 dongs · Jan–Aug 2025 · 320 matched forecasts · target: destination mobility (purposes 4+5+7)", fontsize=10, color="#556c7d")
fig.subplots_adjust(left=.12, right=.97, top=.79, bottom=.26)
fig.savefig(OUT / "search_ablation_wape.png", dpi=180, facecolor="white")
plt.close(fig)
print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
