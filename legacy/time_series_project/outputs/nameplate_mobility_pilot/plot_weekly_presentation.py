"""Create two English slide figures for the weekly pilot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot" / "weekly_40"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def anomaly() -> None:
    panel = pd.read_csv(BASE / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
    panel["week"] = pd.to_datetime(panel.week)
    panel = panel.sort_values(["admin_dong_code", "week"])
    for col in ("y", "subway_exits", "search_age19_39"):
        panel[col + "_yoy"] = panel[col] / panel.groupby("admin_dong_code")[col].shift(52)
    med = panel.groupby("week")[["y_yoy", "subway_exits_yoy", "search_age19_39_yoy"]].median()
    med = med.loc["2024-05-06":"2024-12-30"]
    fig, ax = plt.subplots(figsize=(10.5, 4.7), dpi=180)
    ax.axvspan(pd.Timestamp("2024-08-05"), pd.Timestamp("2024-11-04"), color="#fcd9d1", alpha=.55,
               label="Source anomaly window")
    ax.plot(med.index, med.y_yoy, color="#c6362d", lw=2.7, label="Living mobility, ages 20–39")
    ax.plot(med.index, med.subway_exits_yoy, color="#2764a1", lw=2.2, label="Nearby station exits")
    ax.plot(med.index, med.search_age19_39_yoy, color="#288272", lw=2.2, label="Naver search, ages 19–39")
    ax.axhline(1, color="#888", lw=.9, ls="--")
    ax.set_ylabel("Median ratio to same week in 2023")
    ax.set_title("The 2024 mobility surge is not mirrored by search or transit", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.autofmt_xdate(rotation=0)
    fig.text(.105, .015, "Median across 40 pilot neighborhoods; ratios are descriptive, not causal evidence.", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(OUT / "source_anomaly_english.png", bbox_inches="tight")
    plt.close(fig)


def performance() -> None:
    train = pd.read_csv(BASE / "arima_family" / "validation_metrics.csv").set_index("model")
    test = pd.read_csv(BASE / "arima_family" / "test_metrics.csv").set_index("model")
    blend = pd.read_json(BASE / "ensemble" / "summary.json", typ="series")
    labels = ["Same week\nlast year", "ARIMA", "ARIMAX\n+ search + subway", "Selected\nblend"]
    names = ["lag52", "ARIMA_011", "ARIMAX_011_both"]
    val = [float(train.loc[n, "wape_pct"]) for n in names] + [float(blend.validation_wape)]
    tst = [float(test.loc[n, "wape_pct"]) for n in names] + [float(blend.test_wape)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.7, 4.8), dpi=180)
    bars1 = ax.bar(x - .18, val, .35, color="#95b5d5", label="2024 validation")
    bars2 = ax.bar(x + .18, tst, .35, color="#245789", label="2025 test")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 25)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("A validation-selected blend improves 13-week forecasts", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    fig.text(.11, .015, "40 pilot neighborhoods; 2025 test includes the anomalous 2024 reference period.", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(OUT / "forecast_performance_english.png", bbox_inches="tight")
    plt.close(fig)


def citywide_transfer() -> None:
    table = pd.read_csv(BASE.parent / "weekly_citywide_transfer" / "metrics.csv")
    groups = [("Pilot 40", "pilot40", "full"),
              ("Other 384", "other384", "full"),
              ("All 424\nJan–Jul", "all424", "jan_jul")]
    base, selected = [], []
    for _, scope, period in groups:
        row = table.loc[table.scope.eq(scope) & table.period.eq(period)].iloc[0]
        base.append(row.lag52_wape)
        selected.append(row.selected_blend_wape)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    a = ax.bar(x - .18, base, .35, color="#a9b9c9", label="Same week last year")
    b = ax.bar(x + .18, selected, .35, color="#267b6b", label="Validation-selected blend")
    ax.bar_label(a, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylim(0, 23)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("The selected blend also improves forecasts outside the pilot", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    fig.text(.11, .015, "Full-period bars include Aug 2025, which uses the anomalous Aug 2024 reference.", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(OUT / "citywide_transfer_english.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    anomaly()
    performance()
    citywide_transfer()
    print(OUT / "source_anomaly_english.png")
    print(OUT / "forecast_performance_english.png")
    print(OUT / "citywide_transfer_english.png")
