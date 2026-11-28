"""Transfer the 40-dong validation-selected ARIMA blend to 424 stable dongs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_weekly_arima_family as family

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "weekly_citywide_transfer"
CACHE = OUT / "arima_011_per_dong"
CACHE.mkdir(parents=True, exist_ok=True)


def load_weekly() -> pd.DataFrame:
    files = sorted((BASE / "age2030_daily_baseline").glob("daily_457_age2030_20????.csv"))
    daily = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                       for p in files], ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
    weekly = daily.groupby(["admin_dong_code", "week"], as_index=False).agg(y=("age20_39", "mean"), days=("date", "nunique"))
    weekly = weekly.loc[weekly.days.eq(7)].sort_values(["admin_dong_code", "week"])
    stable = weekly.groupby("admin_dong_code").week.nunique()
    stable = set(stable.index[stable.eq(139)])
    weekly = weekly.loc[weekly.admin_dong_code.isin(stable)].copy()
    if len(stable) != 424 or len(weekly) != 424 * 139:
        raise ValueError("Unexpected stable-dong coverage")
    g = weekly.groupby("admin_dong_code").y
    weekly["lag52"] = g.shift(52)
    weekly["recent4"] = sum(g.shift(13 + j) for j in range(4)) / 4
    weekly["old_recent4"] = sum(g.shift(65 + j) for j in range(4)) / 4
    weekly["growth_ratio"] = (weekly.recent4 / weekly.old_recent4).clip(.8, 1.2)
    weekly["log_y"] = np.log1p(weekly.y)
    return weekly


def score(df: pd.DataFrame, col: str) -> float:
    return round(float(100 * (df.y - df[col]).abs().sum() / df.y.sum()), 3)


def main() -> None:
    weekly = load_weekly()
    codes = sorted(weekly.admin_dong_code.unique())
    pilot = set(pd.read_csv(BASE / "weekly_40" / "weekly_panel_40.csv", dtype={"admin_dong_code": str}).admin_dong_code)
    if not pilot.issubset(codes):
        raise ValueError("Some pilot dongs lack stable citywide history")
    for i, code in enumerate(codes, 1):
        target = CACHE / f"{code}.csv"
        if not target.exists():
            dong = weekly.loc[weekly.admin_dong_code.eq(code)].copy()
            pred, _ = family.forecast_one(dong, "test", "ARIMA_011")
            pred = pred.rename(columns={"predicted": "arima_pred"})[["admin_dong_code", "week", "arima_pred"]]
            pred.to_csv(target, index=False, encoding="utf-8-sig")
        if i % 25 == 0 or i == len(codes):
            print(f"citywide ARIMA: {i}/{len(codes)} dongs", flush=True)
    arima = pd.concat([pd.read_csv(CACHE / f"{code}.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
                       for code in codes], ignore_index=True)
    test = weekly.loc[weekly.week.between("2025-01-06", "2025-08-25")].copy()
    test["week"] = test.week.dt.strftime("%Y-%m-%d")
    test = test.merge(arima, on=["admin_dong_code", "week"], validate="one_to_one")
    if len(test) != 424 * 34 or test[["y", "lag52", "growth_ratio", "arima_pred"]].isna().any().any():
        raise ValueError("Incomplete citywide forecast panel")
    selection = json.loads((BASE / "weekly_40" / "ensemble" / "summary.json").read_text(encoding="utf-8"))
    power, weight = selection["selected_growth_power"], selection["selected_arima_weight"]
    test["growth_corrected"] = test.lag52 * test.growth_ratio.pow(power)
    test["selected_blend"] = (1 - weight) * test.growth_corrected + weight * test.arima_pred
    test["prior_year_week"] = (pd.to_datetime(test.week) - pd.Timedelta(weeks=52)).dt.strftime("%Y-%m-%d")
    flag = test.prior_year_week.between("2024-08-05", "2024-10-28")
    test["quality_guard_sensitivity"] = test.selected_blend.where(~flag, test.arima_pred)
    test["pilot40"] = test.admin_dong_code.isin(pilot)
    test.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = []
    for scope, part in (("all424", test), ("pilot40", test.loc[test.pilot40]),
                        ("other384", test.loc[~test.pilot40])):
        for period, data in (("full", part), ("jan_jul", part.loc[part.week.lt("2025-08-01")])):
            metrics.append({"scope": scope, "period": period, "dongs": data.admin_dong_code.nunique(), "n": len(data),
                            **{name + "_wape": score(data, name) for name in
                               ("lag52", "arima_pred", "growth_corrected", "selected_blend", "quality_guard_sensitivity")}})
    table = pd.DataFrame(metrics)
    table.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
