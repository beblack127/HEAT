"""Summarize ACF/PACF heterogeneity across all 424 complete-history dongs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from run_weekly_citywide_transfer import load_weekly

OUT = Path(__file__).resolve().parent / "mobility_only_dong"
OUT.mkdir(exist_ok=True)

rows = []
for code, g in load_weekly().groupby("admin_dong_code"):
    level = np.log1p(g.sort_values("week").y.to_numpy())
    diff = np.diff(level)
    a = acf(level, nlags=52, fft=True)
    pa = pacf(level, nlags=13, method="ywm")
    ad = acf(diff, nlags=52, fft=True)
    pdiff = pacf(diff, nlags=13, method="ywm")
    rows.append({"admin_dong_code": code,
                 "acf_level_lag1": a[1], "acf_level_lag13": a[13], "acf_level_lag52": a[52],
                 "pacf_level_lag1": pa[1], "pacf_level_lag13": pa[13],
                 "acf_diff_lag1": ad[1], "acf_diff_lag13": ad[13], "acf_diff_lag52": ad[52],
                 "pacf_diff_lag1": pdiff[1], "pacf_diff_lag13": pdiff[13]})
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "all424_acf_pacf_selected_lags.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180)
sets = [
    ("ACF | log level", ["acf_level_lag1", "acf_level_lag13", "acf_level_lag52"]),
    ("PACF | log level", ["pacf_level_lag1", "pacf_level_lag13"]),
    ("ACF | first difference", ["acf_diff_lag1", "acf_diff_lag13", "acf_diff_lag52"]),
    ("PACF | first difference", ["pacf_diff_lag1", "pacf_diff_lag13"]),
]
for ax, (title, cols) in zip(axes.flat, sets):
    ax.boxplot([frame[c].to_numpy() for c in cols], tick_labels=[c.split("lag")[-1] for c in cols],
               patch_artist=True, boxprops={"facecolor": "#dbe8f5", "edgecolor": "#3368a6"},
               medianprops={"color": "#d3493b", "linewidth": 2}, showfliers=False)
    ax.axhline(0, color="#7a8797", lw=.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("Correlation")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ACF/PACF across 424 neighborhoods | purpose 4+5+7, ages 20s–30s", fontsize=14)
fig.text(.08, .005, "Each box summarizes 424 individual neighborhood time series (139 complete weeks).\n"
         "2024 Aug–Oct source-level surge is present; these distributions do not select one model order for every dong.",
         fontsize=8.5, color="#596675")
fig.tight_layout(rect=(0, .06, 1, .95))
fig.savefig(OUT / "all424_acf_pacf_distribution.png", bbox_inches="tight")
print(frame.drop(columns="admin_dong_code").median().round(3).to_string())
