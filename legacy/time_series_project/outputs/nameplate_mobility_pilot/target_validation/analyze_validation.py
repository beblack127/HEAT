"""Diagnostics and figure for mobility-target construct validation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "yoy_change_40.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
signals = ["mobility_457_daily", "mobility_all_daily", "mobility_45_daily"]
outcomes = ["footfall", "food_transactions"]

def rank_corr(frame: pd.DataFrame, signal: str, outcome: str) -> float:
    a = frame[signal + "_yoy_log_demeaned"]
    b = frame[outcome + "_yoy_log_demeaned"]
    return float(spearmanr(a, b).statistic)

rng = np.random.default_rng(42)
dongs = df.admin_dong_code.unique()
ordered = df.sort_values(["admin_dong_code", "quarter"])
boot = {}
for outcome in outcomes:
    boot[outcome] = {}
    y_by_dong = np.vstack([ordered.loc[ordered.admin_dong_code.eq(code), outcome + "_yoy_log_demeaned"].to_numpy()
                           for code in dongs])
    for signal in signals:
        x_by_dong = np.vstack([ordered.loc[ordered.admin_dong_code.eq(code), signal + "_yoy_log_demeaned"].to_numpy()
                               for code in dongs])
        draws = []
        for _ in range(1500):
            sampled = rng.integers(0, len(dongs), size=len(dongs))
            draws.append(float(spearmanr(x_by_dong[sampled].ravel(), y_by_dong[sampled].ravel()).statistic))
        boot[outcome][signal] = {
            "rho": round(rank_corr(df, signal, outcome), 3),
            "dong_bootstrap_95pct": [round(float(x), 3) for x in np.quantile(draws, [.025, .975])],
        }

topk = {}
for outcome in outcomes:
    topk[outcome] = {}
    for signal in signals:
        by_quarter = {}
        for q, group in df.groupby("quarter"):
            a = set(group.nlargest(10, signal + "_yoy_log").admin_dong_code)
            b = set(group.nlargest(10, outcome + "_yoy_log").admin_dong_code)
            by_quarter[str(q)] = len(a & b)
        topk[outcome][signal] = {"hits_out_of_40": sum(by_quarter.values()), "by_quarter": by_quarter}

result = {
    "method": "4 year-over-year quarters × 40 dongs; quarter-demeaned log changes; dong cluster bootstrap; exploratory",
    "bootstrap": boot,
    "top10_growth_overlap": topk,
    "same_method_2025_yoy_only": {
        outcome: {
            signal: {"rho": round(rank_corr(df.loc[df.quarter.isin([20251, 20252])], signal, outcome), 3),
                     "n_dong_quarters": 80}
            for signal in signals}
        for outcome in outcomes
    },
}
subset = df.loc[df.quarter.isin([20251, 20252])].sort_values(["admin_dong_code", "quarter"])
for outcome in outcomes:
    y_two = np.vstack([subset.loc[subset.admin_dong_code.eq(code), outcome + "_yoy_log_demeaned"].to_numpy()
                       for code in dongs])
    for signal in signals:
        x_two = np.vstack([subset.loc[subset.admin_dong_code.eq(code), signal + "_yoy_log_demeaned"].to_numpy()
                           for code in dongs])
        draws = []
        for _ in range(1500):
            sampled = rng.integers(0, len(dongs), size=len(dongs))
            draws.append(float(spearmanr(x_two[sampled].ravel(), y_two[sampled].ravel()).statistic))
        result["same_method_2025_yoy_only"][outcome][signal]["dong_bootstrap_95pct"] = [
            round(float(x), 3) for x in np.quantile(draws, [.025, .975])
        ]
(OUT / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

plot_df = df.loc[df.quarter.isin([20251, 20252])]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor="white")
colors = {20251: "#0c8187", 20252: "#db9436"}
labels = {"footfall": "Official dong footfall estimate", "food_transactions": "Allocated food sales transactions"}
for ax, outcome in zip(axes, outcomes):
    for q, g in plot_df.groupby("quarter"):
        ax.scatter(g["mobility_457_daily_yoy_log_demeaned"], g[outcome + "_yoy_log_demeaned"],
                   s=38, alpha=.78, color=colors[int(q)], label=f"{str(q)[:4]} Q{str(q)[-1]}")
    ax.axhline(0, color="#b8c6ce", linewidth=1)
    ax.axvline(0, color="#b8c6ce", linewidth=1)
    ax.set_title(f"{labels[outcome]}\nSpearman ρ = {result['same_method_2025_yoy_only'][outcome]['mobility_457_daily']['rho']:.2f}",
                 loc="left", fontsize=13, fontweight="bold", color="#163047")
    ax.set_xlabel("Mobility 4+5+7: YoY log change, quarter-adjusted", fontsize=10)
    ax.grid(alpha=.10)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_ylabel("External indicator: YoY log change, quarter-adjusted", fontsize=10)
axes[1].legend(title="Quarter", frameon=False, fontsize=9)
fig.suptitle("Does the mobility target track commercial activity?", x=.07, ha="left",
             fontsize=20, fontweight="bold", color="#163047")
fig.text(.07, .02, "40 dongs × 2 YoY quarters (2025 Q1–Q2 vs 2024). External measures are estimates; correlations are descriptive.",
         fontsize=9, color="#526a7d")
fig.subplots_adjust(left=.07, right=.97, top=.76, bottom=.19, wspace=.25)
fig.savefig(OUT / "target_validation_scatter.png", dpi=180, facecolor="white")
plt.close(fig)
print(json.dumps(result, ensure_ascii=False, indent=2))
