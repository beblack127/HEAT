"""Select a simple seasonal-naive/ARIMA blend on 2024 validation only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "weekly_40" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)
ARIMA = BASE / "weekly_40" / "arima_family"


def panel() -> pd.DataFrame:
    frame = pd.read_csv(BASE / "weekly_40" / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
    frame = frame.sort_values(["admin_dong_code", "week"])
    g = frame.groupby("admin_dong_code").y
    frame["lag52"] = g.shift(52)
    frame["recent4_origin"] = sum(g.shift(13 + i) for i in range(4)) / 4
    frame["recent4_year_before_origin"] = sum(g.shift(65 + i) for i in range(4)) / 4
    frame["growth_ratio"] = (frame.recent4_origin / frame.recent4_year_before_origin).clip(.8, 1.2)
    return frame


def fold_data(name: str, weekly: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(ARIMA / f"{name}_predictions.csv", dtype={"admin_dong_code": str})
    pred["week"] = pred.week.str[:10]
    arima = pred.loc[pred.model.eq("ARIMA_011"), ["admin_dong_code", "week", "predicted"]]
    arima = arima.rename(columns={"predicted": "arima_pred"})
    base = pred.loc[pred.model.eq("lag52"), ["admin_dong_code", "week", "actual", "predicted"]]
    base = base.rename(columns={"predicted": "lag52_pred"})
    combined = base.merge(arima, on=["admin_dong_code", "week"], validate="one_to_one")
    combined = combined.merge(weekly[["admin_dong_code", "week", "growth_ratio"]],
                              on=["admin_dong_code", "week"], validate="one_to_one")
    if combined[["actual", "lag52_pred", "arima_pred", "growth_ratio"]].isna().any().any():
        raise ValueError("Missing ensemble input")
    return combined


def score(actual: pd.Series, predicted: pd.Series) -> float:
    return float(100 * (actual - predicted).abs().sum() / actual.sum())


def main() -> None:
    weekly = panel()
    validation = fold_data("validation", weekly)
    test = fold_data("test", weekly)
    grid = []
    for growth_power in (0, .25, .5):
        for arima_weight in (0, .25, .5):
            seasonal = validation.lag52_pred * validation.growth_ratio.pow(growth_power)
            predicted = seasonal * (1 - arima_weight) + validation.arima_pred * arima_weight
            grid.append({"growth_power": growth_power, "arima_weight": arima_weight,
                         "validation_wape": round(score(validation.actual, predicted), 3)})
    grid = pd.DataFrame(grid).sort_values(["validation_wape", "growth_power", "arima_weight"])
    grid.to_csv(OUT / "validation_grid.csv", index=False, encoding="utf-8-sig")
    chosen = grid.iloc[0]
    for fold_name, df in (("validation", validation), ("test", test)):
        df = df.copy()
        df["selected_pred"] = (df.lag52_pred * df.growth_ratio.pow(chosen.growth_power) * (1 - chosen.arima_weight)
                               + df.arima_pred * chosen.arima_weight)
        df.to_csv(OUT / f"{fold_name}_predictions.csv", index=False, encoding="utf-8-sig")
    test = pd.read_csv(OUT / "test_predictions.csv", dtype={"admin_dong_code": str})
    test["month"] = test.week.str[:7]
    results = {
        "selected_on": "2024-05-06 to 2024-07-29 validation only",
        "selected_growth_power": float(chosen.growth_power),
        "selected_arima_weight": float(chosen.arima_weight),
        "validation_wape": float(chosen.validation_wape),
        "test_wape": round(score(test.actual, test.selected_pred), 3),
        "test_lag52_wape": round(score(test.actual, test.lag52_pred), 3),
        "test_arima_wape": round(score(test.actual, test.arima_pred), 3),
        "test_jan_jul_wape": round(score(test.loc[test.month.lt("2025-08"), "actual"],
                                          test.loc[test.month.lt("2025-08"), "selected_pred"]), 3),
        "test_jan_jul_lag52_wape": round(score(test.loc[test.month.lt("2025-08"), "actual"],
                                                test.loc[test.month.lt("2025-08"), "lag52_pred"]), 3),
    }
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
