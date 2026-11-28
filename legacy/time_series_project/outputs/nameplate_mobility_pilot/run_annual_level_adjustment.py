"""Exploratory 13-week fixed-origin test of a known-at-origin yearly adjustment."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "annual_level_adjustment"
OUT.mkdir(exist_ok=True)
SEARCH = (BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
          "search" / "행정동_검색지수_월별.csv")
ORIGINS = ["2024-10-27", "2025-01-26", "2025-04-27"]


def wape(frame: pd.DataFrame) -> float:
    return 100 * (frame.actual - frame.predicted).abs().sum() / frame.actual.sum()


files = sorted((BASE / "age2030_daily_baseline").glob("daily_457_age2030_20????.csv"))
daily = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                   for f in files], ignore_index=True)
daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
weekly = daily.groupby(["admin_dong_code", "week"], as_index=False).agg(
    y=("age20_39", "mean"), days=("date", "nunique"))
weekly = weekly.loc[weekly.days.eq(7)].sort_values(["admin_dong_code", "week"])
stable = weekly.groupby("admin_dong_code").week.nunique()
weekly = weekly.loc[weekly.admin_dong_code.isin(stable.loc[stable.eq(139)].index)].copy()
search = pd.read_csv(SEARCH, encoding="utf-8-sig", dtype={"dong_code8": str})
codes = set(search.loc[search.tier.eq("전체"), "dong_code8"]) & set(weekly.admin_dong_code)
weekly = weekly.loc[weekly.admin_dong_code.isin(codes)].copy()
lookup = weekly.set_index(["admin_dong_code", "week"])["y"]
rows = []
for origin_str in ORIGINS:
    origin = pd.Timestamp(origin_str)
    for code in sorted(codes):
        # Four complete Monday-Sunday weeks ending at the forecast origin.
        current_weeks = pd.date_range(origin - pd.Timedelta(weeks=3, days=6),
                                      origin - pd.Timedelta(days=6), freq="7D")
        future_weeks = pd.date_range(origin + pd.Timedelta(days=1), periods=13, freq="7D")
        current = np.array([lookup.loc[(code, week)] for week in current_weeks])
        previous = np.array([lookup.loc[(code, week - pd.Timedelta(weeks=52))]
                             for week in current_weeks])
        ratio = np.clip(current.mean() / previous.mean(), 0.5, 1.5)
        for step, week in enumerate(future_weeks, 1):
            actual = lookup.loc[(code, week)]
            yearly = lookup.loc[(code, week - pd.Timedelta(weeks=52))]
            for name, multiplier in [("Same week last year", 1.0),
                                      ("Yearly + recent level", ratio ** 0.5)]:
                rows.append({"origin": origin_str, "admin_dong_code": code,
                             "week": week.date().isoformat(), "step": step,
                             "model": name, "ratio": ratio, "actual": actual,
                             "predicted": yearly * multiplier})

pred = pd.DataFrame(rows)
pred.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
summary = pred.groupby(["origin", "model"]).apply(wape, include_groups=False).reset_index(name="wape_pct")
summary.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
case = pred.loc[pred.admin_dong_code.isin(["11170640", "11350665", "11470640", "11500540"])].copy()
case.groupby(["origin", "admin_dong_code", "model"]).apply(wape, include_groups=False).reset_index(name="wape_pct").to_csv(
    OUT / "case_metrics.csv", index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))
