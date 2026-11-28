"""Render the existing four-pair forecast comparison with all 108 fitting weeks."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from run_weekly_citywide_transfer import load_weekly


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "fixed_origin_search"
OUT = SOURCE / "full_history_figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

old = pd.read_csv(BASE / "fixed_origin_3month" / "backtest_predictions.csv",
                  dtype={"admin_dong_code": str})
new = pd.read_csv(SOURCE / "backtest_search_predictions.csv",
                  dtype={"admin_dong_code": str})
predictions = pd.concat([old, new], ignore_index=True)
predictions["week"] = pd.to_datetime(predictions.week)
weekly = load_weekly()
names = pd.read_csv(BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
                    "nametags" / "01_행정동_목록.csv", encoding="utf-8-sig",
                    dtype={"dong_code8": str})
names = names.drop_duplicates("dong_code8").set_index("dong_code8")

pairs = [
    ("ARIMA family", "ARIMA", "ARIMAX_search"),
    ("Seasonal ARIMA family", "SARIMA", "SARIMAX_search"),
    ("ARIMAX + lagged mobility", "ARIMAX", "ARIMAX_search+mobility"),
    ("SARIMAX + lagged mobility", "SARIMAX", "SARIMAX_search+mobility"),
]
origin_week = pd.Timestamp("2025-04-21")
start_week = pd.Timestamp("2023-04-03")
cases = [
    ("11500540", "best"),
    ("11470640", "best"),
    ("11350665", "worst"),
    ("11170640", "worst"),
]

for code, category in cases:
    area = f"{names.loc[code, 'gu']} {names.loc[code, 'dong_name']}"
    history = weekly.loc[weekly.admin_dong_code.eq(code) &
                         weekly.week.between(start_week, origin_week)]
    future = weekly.loc[weekly.admin_dong_code.eq(code) &
                        weekly.week.gt(origin_week) &
                        weekly.week.le(origin_week + pd.Timedelta(weeks=13))]
    if len(history) != 108 or len(future) != 13:
        raise ValueError(f"Unexpected observed history for {code}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True, dpi=180)
    for ax, (label, without, with_search) in zip(axes.flat, pairs):
        no = predictions.loc[predictions.admin_dong_code.eq(code) &
                             predictions.model.eq(without)].sort_values("week")
        yes = predictions.loc[predictions.admin_dong_code.eq(code) &
                              predictions.model.eq(with_search)].sort_values("week")
        if len(no) != 13 or len(yes) != 13:
            raise ValueError(f"Incomplete forecast {code} {label}")
        wape_no = 100 * (no.actual - no.predicted).abs().sum() / no.actual.sum()
        wape_yes = 100 * (yes.actual - yes.predicted).abs().sum() / yes.actual.sum()
        ax.plot(history.week, history.y, color="#3368a6", lw=1.8, label="Observed by cutoff")
        ax.plot(future.week, future.y, color="#3368a6", ls="--", lw=1.6, label="Later observed")
        ax.plot(no.week, no.predicted, color="#7d8792", lw=1.6,
                label=f"No search: {wape_no:.1f}%")
        ax.plot(yes.week, yes.predicted, color="#d3493b", lw=1.8,
                label=f"With search: {wape_yes:.1f}%")
        ax.axvline(origin_week, color="#7a8797", ls=":", lw=1.1)
        ax.set_title(label, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"{area} ({code}) | full fitting history and fixed-origin 13-week forecast",
                 fontsize=16)
    fig.text(.07, .005,
             "Fitting target: 2023-04-03 to 2025-04-27 (108 weeks). Forecast: 2025-04-28 to 2025-07-27. Case selected after evaluation.",
             fontsize=9, color="#596675")
    fig.tight_layout(rect=(0, .025, 1, .94))
    path = OUT / f"{category}_{code}_four_pairs_from_2023_04.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)
