"""Produce presentation-ready audits for weekly forecasts and source anomaly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
WEEKLY = BASE / "weekly_40"
OUT = WEEKLY / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    return float(100 * (actual - predicted).abs().sum() / actual.sum())


def anomaly_audit(weekly: pd.DataFrame) -> dict:
    weekly = weekly.sort_values(["admin_dong_code", "week"]).copy()
    for col in ("y", "subway_exits", "search_all", "search_age19_39"):
        weekly[col + "_yoy"] = weekly[col] / weekly.groupby("admin_dong_code")[col].shift(52)
    periods = {"pre": ("2024-05-06", "2024-07-29"),
               "spike": ("2024-08-05", "2024-10-28"),
               "post": ("2024-11-04", "2024-12-30")}
    result = {}
    for name, (start, end) in periods.items():
        part = weekly.loc[weekly.week.between(start, end)]
        result[name] = {col: round(float(part[col + "_yoy"].median()), 3)
                        for col in ("y", "subway_exits", "search_all", "search_age19_39")}
    return result


def citywide_baselines() -> dict:
    files = sorted((BASE / "age2030_daily_baseline").glob("daily_457_age2030_20????.csv"))
    daily = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                       for p in files], ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
    weekly = daily.groupby(["admin_dong_code", "week"], as_index=False).agg(y=("age20_39", "mean"), days=("date", "nunique"))
    weekly = weekly.loc[weekly.days.eq(7)].sort_values(["admin_dong_code", "week"])
    g = weekly.groupby("admin_dong_code").y
    weekly["lag13"] = g.shift(13)
    weekly["lag52"] = g.shift(52)
    weekly["recent4"] = sum(g.shift(13 + i) for i in range(4)) / 4
    weekly["old_recent4"] = sum(g.shift(65 + i) for i in range(4)) / 4
    test = weekly.loc[weekly.week.between("2025-01-06", "2025-08-25")].dropna().copy()
    test["growth_corrected"] = test.lag52 * (test.recent4 / test.old_recent4).clip(.8, 1.2).pow(.25)
    test["prior_year_week"] = test.week - pd.Timedelta(weeks=52)
    # Sensitivity only: known 2024 source-anomaly interval, never replace GT.
    anomalous_reference = test.prior_year_week.between("2024-08-05", "2024-10-28")
    test["quality_guard"] = test.growth_corrected.where(~anomalous_reference, test.lag13)
    test.to_csv(OUT / "citywide_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    jan_jul = test.loc[test.week.lt("2025-08-01")]
    return {"n": len(test), "dongs": int(test.admin_dong_code.nunique()),
            "full": {col: round(wape(test.y, test[col]), 2) for col in ("lag13", "lag52", "growth_corrected", "quality_guard")},
            "jan_jul": {col: round(wape(jan_jul.y, jan_jul[col]), 2) for col in ("lag13", "lag52", "growth_corrected")},
            "quality_guard_note": "2024-08-05 to 2024-10-28 reference weeks switch to lag13; exploratory sensitivity"}


def pilot_diagnostics() -> dict:
    test = pd.read_csv(WEEKLY / "ensemble" / "test_predictions.csv", dtype={"admin_dong_code": str})
    test["week"] = pd.to_datetime(test.week)
    test["prior_year_week"] = test.week - pd.Timedelta(weeks=52)
    flagged = test.prior_year_week.between("2024-08-05", "2024-10-28")
    test["quality_guard"] = test.selected_pred.where(~flagged, test.arima_pred)
    test["selected_abs_error"] = (test.actual - test.selected_pred).abs()
    test["lag52_abs_error"] = (test.actual - test.lag52_pred).abs()
    test["month"] = test.week.dt.strftime("%Y-%m")
    test.to_csv(OUT / "pilot_40_diagnostics.csv", index=False, encoding="utf-8-sig")
    per_dong = test.groupby("admin_dong_code").agg(actual=("actual", "sum"),
                                                     selected_abs_error=("selected_abs_error", "sum"),
                                                     lag52_abs_error=("lag52_abs_error", "sum"))
    per_dong["wape_gain_pp"] = 100 * (per_dong.lag52_abs_error - per_dong.selected_abs_error) / per_dong.actual
    per_dong.to_csv(OUT / "per_dong_wape.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(42)
    v = per_dong[["selected_abs_error", "lag52_abs_error", "actual"]].to_numpy()
    draws = v[rng.integers(0, len(v), size=(5000, len(v)))].sum(axis=1)
    gain = 100 * (draws[:, 1] - draws[:, 0]) / draws[:, 2]
    monthly = test.groupby("month").apply(lambda x: pd.Series({
        "selected_wape": round(wape(x.actual, x.selected_pred), 2),
        "lag52_wape": round(wape(x.actual, x.lag52_pred), 2),
        "guard_wape": round(wape(x.actual, x.quality_guard), 2),
    }), include_groups=False)
    monthly.to_csv(OUT / "monthly_wape.csv", encoding="utf-8-sig")
    rank = test.sort_values(["admin_dong_code", "week"]).copy()
    for col in ("actual", "lag52_pred", "arima_pred", "selected_pred"):
        rank[col + "_4w"] = rank.groupby("admin_dong_code")[col].transform(lambda s: s.rolling(4).mean())
    rank = rank.dropna(subset=[c + "_4w" for c in ("actual", "lag52_pred", "arima_pred", "selected_pred")])
    rank["actual_growth"] = rank.actual_4w / rank.lag52_pred_4w - 1
    top10 = {}
    for model in ("arima_pred", "selected_pred"):
        rank["pred_growth"] = rank[model + "_4w"] / rank.lag52_pred_4w - 1
        hits = []
        for week, group in rank.groupby("week"):
            actual_top = set(group.nlargest(10, "actual_growth").admin_dong_code)
            predicted_top = set(group.nlargest(10, "pred_growth").admin_dong_code)
            hits.append(len(actual_top & predicted_top))
        top10[model] = {"hits": sum(hits), "out_of": 10 * len(hits)}
    return {"selected_wape": round(wape(test.actual, test.selected_pred), 2),
            "lag52_wape": round(wape(test.actual, test.lag52_pred), 2),
            "quality_guard_wape": round(wape(test.actual, test.quality_guard), 2),
            "dongs_improved": int((per_dong.selected_abs_error < per_dong.lag52_abs_error).sum()),
            "dong_cluster_bootstrap_gain_pp_95pct": [round(float(x), 2) for x in np.quantile(gain, [.025, .975])],
            "growth_top10_4week": top10,
            "rank_note": "post-hoc exploratory ranking on selected 40 dongs; rolling 4-week YoY growth"}


if __name__ == "__main__":
    weekly = pd.read_csv(WEEKLY / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
    report = {"anomaly_external_check": anomaly_audit(weekly),
              "citywide_baselines": citywide_baselines(),
              "pilot_40": pilot_diagnostics()}
    (OUT / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
