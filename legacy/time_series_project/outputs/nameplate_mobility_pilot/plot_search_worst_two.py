"""Show four matched no-search/search model pairs for the two worst-fit dongs."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from run_weekly_citywide_transfer import load_weekly

BASE = Path(__file__).resolve().parent
OUT = BASE / "fixed_origin_search"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

old = pd.read_csv(BASE / "fixed_origin_3month" / "backtest_predictions.csv",
                  dtype={"admin_dong_code": str})
new = pd.read_csv(OUT / "backtest_search_predictions.csv",
                  dtype={"admin_dong_code": str})
all_pred = pd.concat([old, new], ignore_index=True)
all_pred["week"] = pd.to_datetime(all_pred.week)
weekly = load_weekly()
names = pd.read_csv(BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
                    "nametags" / "01_행정동_목록.csv", encoding="utf-8-sig",
                    dtype={"dong_code8": str})
name_map = names.drop_duplicates("dong_code8").set_index("dong_code8")["dong_name"].to_dict()
gu_map = names.drop_duplicates("dong_code8").set_index("dong_code8")["gu"].to_dict()

rank = new.loc[new.model.eq("SARIMAX_search+mobility")].groupby("admin_dong_code").apply(
    lambda g: 100 * (g.actual - g.predicted).abs().sum() / g.actual.sum(),
    include_groups=False).sort_values()
chosen = rank.tail(2).sort_values(ascending=False)
if len(chosen) != 2:
    raise ValueError("Expected two worst-fit dongs")

pairs = [
    ("ARIMA family", "ARIMA", "ARIMAX_search"),
    ("Seasonal ARIMA family", "SARIMA", "SARIMAX_search"),
    ("ARIMAX + lagged mobility", "ARIMAX", "ARIMAX_search+mobility"),
    ("SARIMAX + lagged mobility", "SARIMAX", "SARIMAX_search+mobility"),
]
selection = []
comparison_rows = []
origin = pd.Timestamp("2025-04-21")
for code, selected_wape in chosen.items():
    area = f"{gu_map.get(code, '')} {name_map.get(code, '')}"
    selection.append({"admin_dong_code": code, "area": area,
                      "selection_model": "SARIMAX_search+mobility",
                      "selection_wape_pct": round(float(selected_wape), 3),
                      "rank": len(selection) + 1})
    history = weekly.loc[weekly.admin_dong_code.eq(code) &
                         weekly.week.between(origin - pd.Timedelta(weeks=20), origin)]
    actual = weekly.loc[weekly.admin_dong_code.eq(code) &
                        weekly.week.gt(origin) & weekly.week.le(origin + pd.Timedelta(weeks=13))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True, dpi=180)
    for ax, (label, no_name, yes_name) in zip(axes.flat, pairs):
        no = all_pred.loc[all_pred.admin_dong_code.eq(code) & all_pred.model.eq(no_name)].sort_values("week")
        yes = all_pred.loc[all_pred.admin_dong_code.eq(code) & all_pred.model.eq(yes_name)].sort_values("week")
        if len(no) != 13 or len(yes) != 13 or not (no.week.to_numpy() == yes.week.to_numpy()).all():
            raise ValueError(f"Mismatched pair {code} {label}")
        wape_no = 100 * (no.actual - no.predicted).abs().sum() / no.actual.sum()
        wape_yes = 100 * (yes.actual - yes.predicted).abs().sum() / yes.actual.sum()
        comparison_rows.append({"admin_dong_code": code, "area": area, "pair": label,
                                "no_search_model": no_name, "search_model": yes_name,
                                "wape_no_search_pct": round(float(wape_no), 3),
                                "wape_search_pct": round(float(wape_yes), 3),
                                "search_gain_pp": round(float(wape_no - wape_yes), 3)})
        ax.plot(history.week, history.y, color="#3368a6", lw=1.8, label="Observed by cutoff")
        ax.plot(actual.week, actual.y, color="#3368a6", ls="--", lw=1.6, label="Later observed")
        ax.plot(no.week, no.predicted, color="#7d8792", lw=1.6,
                label=f"No search: {wape_no:.1f}%")
        ax.plot(yes.week, yes.predicted, color="#d3493b", lw=1.8,
                label=f"With search: {wape_yes:.1f}%")
        ax.axvline(origin, color="#7a8797", ls=":", lw=1.1)
        ax.set_title(label, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"{area} ({code}) | fixed-origin 13-week forecast", fontsize=16)
    fig.text(.07, .005, "Cutoff: 2025-04-27. Selected after evaluation by highest SARIMAX + search + mobility WAPE; illustrative only.",
             fontsize=9, color="#596675")
    fig.tight_layout(rect=(0, .025, 1, .94))
    fig.savefig(OUT / f"worst_fit_{code}_four_pairs.png", bbox_inches="tight")
    plt.close(fig)

pd.DataFrame(selection).to_csv(OUT / "worst_two_selection.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(comparison_rows).to_csv(OUT / "worst_two_model_pairs.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(selection).to_string(index=False))
