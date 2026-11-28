"""Apply the existing citywide ARIMA blend to the 2024 validation period."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_weekly_arima_family as family
from run_weekly_citywide_transfer import load_weekly

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "weekly_citywide_transfer"
CACHE = OUT / "validation_arima_011_per_dong"
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> None:
    weekly = load_weekly()
    codes = sorted(weekly.admin_dong_code.unique())
    for i, code in enumerate(codes, 1):
        target = CACHE / f"{code}.csv"
        if not target.exists():
            dong = weekly.loc[weekly.admin_dong_code.eq(code)]
            predicted, _ = family.forecast_one(dong, "validation", "ARIMA_011")
            predicted = predicted.rename(columns={"predicted": "arima_pred"})[["admin_dong_code", "week", "arima_pred"]]
            predicted.to_csv(target, index=False, encoding="utf-8-sig")
        if i % 50 == 0 or i == len(codes):
            print(f"citywide validation: {i}/{len(codes)}", flush=True)
    predictions = pd.concat([pd.read_csv(CACHE / f"{code}.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
                             for code in codes], ignore_index=True)
    val = weekly.loc[weekly.week.between("2024-05-06", "2024-07-29")].copy()
    val["week"] = val.week.dt.strftime("%Y-%m-%d")
    val = val.merge(predictions, on=["admin_dong_code", "week"], validate="one_to_one")
    selection = json.loads((BASE / "weekly_40" / "ensemble" / "summary.json").read_text(encoding="utf-8"))
    power, weight = selection["selected_growth_power"], selection["selected_arima_weight"]
    val["growth_corrected"] = val.lag52 * val.growth_ratio.pow(power)
    val["selected_blend"] = (1 - weight) * val.growth_corrected + weight * val.arima_pred
    if len(val) != 424 * 13 or val[["y", "lag52", "arima_pred", "selected_blend"]].isna().any().any():
        raise ValueError("Incomplete validation panel")
    val.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {len(val)} validation rows", flush=True)


if __name__ == "__main__":
    main()
