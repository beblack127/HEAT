"""Evaluate 13-week-ahead emerging-place ranking from living-mobility forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_weekly_citywide_transfer import load_weekly

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
SOURCE = BASE / "weekly_citywide_transfer"
OUT = BASE / "weekly_hotplace_rank"
OUT.mkdir(parents=True, exist_ok=True)
K = 20


def volume_floor() -> float:
    weekly = load_weekly().sort_values(["admin_dong_code", "week"])
    weekly["y4"] = weekly.groupby("admin_dong_code").y.transform(lambda s: s.rolling(4).mean())
    return float(weekly.loc[weekly.week.lt("2024-01-01"), "y4"].quantile(.60))


def prepare(fold: str, floor: float) -> pd.DataFrame:
    file = "validation_predictions.csv" if fold == "validation" else "predictions.csv"
    frame = pd.read_csv(SOURCE / file, encoding="utf-8-sig", dtype={"admin_dong_code": str})
    frame = frame.sort_values(["admin_dong_code", "week"])
    for col in ("y", "lag52", "recent4", "arima_pred", "selected_blend"):
        frame[col + "_4w"] = frame.groupby("admin_dong_code")[col].transform(lambda s: s.rolling(4).mean())
    frame = frame.dropna(subset=[col + "_4w" for col in ("y", "lag52", "recent4", "arima_pred", "selected_blend")])
    if fold == "validation":
        frame = frame.loc[frame.week.lt("2024-07-29")].copy()  # source-anomaly onset
    else:
        frame = frame.loc[frame.week.lt("2025-08-01")].copy()  # prior-year source-anomaly reference
    for col in ("y", "lag52", "recent4", "arima_pred", "selected_blend"):
        frame[col + "_share"] = frame[col + "_4w"] / frame.groupby("week")[col + "_4w"].transform("sum")
    frame["actual_share_growth"] = frame.y_share / frame.lag52_share - 1
    for model in ("recent4", "arima_pred", "selected_blend"):
        frame[model + "_share_growth"] = frame[model + "_share"] / frame.lag52_share - 1
    frame["volume_floor"] = floor
    return frame


def rank(fold: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, group in frame.groupby("week"):
        actual = set(group.loc[group.y_4w.ge(group.volume_floor)].nlargest(K, "actual_share_growth").admin_dong_code)
        if len(actual) != K:
            raise ValueError("Too few actual candidates")
        for model in ("recent4", "arima_pred", "selected_blend"):
            predicted = set(group.loc[group[model + "_4w"].ge(group.volume_floor)].nlargest(K, model + "_share_growth").admin_dong_code)
            if len(predicted) != K:
                raise ValueError("Too few predicted candidates")
            rows.append({"fold": fold, "week": week, "model": model,
                         "hits_at_20": len(actual & predicted),
                         "precision_at_20": len(actual & predicted) / K,
                         "actual_codes": ",".join(sorted(actual)),
                         "predicted_codes": ",".join(sorted(predicted))})
    return pd.DataFrame(rows)


def main() -> None:
    floor = volume_floor()
    validation = prepare("validation", floor)
    test = prepare("test", floor)
    validation.to_csv(OUT / "validation_panel.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT / "test_panel.csv", index=False, encoding="utf-8-sig")
    val_rank = rank("validation", validation)
    test_rank = rank("test", test)
    val_rank.to_csv(OUT / "validation_weekly_rank.csv", index=False, encoding="utf-8-sig")
    test_rank.to_csv(OUT / "test_weekly_rank.csv", index=False, encoding="utf-8-sig")
    val_scores = val_rank.groupby("model").hits_at_20.sum().sort_values(ascending=False)
    chosen = val_scores.index[0]
    test_scores = test_rank.groupby("model").hits_at_20.sum().sort_values(ascending=False)
    summary = {"target": "top 20 in four-week relative Seoul-share growth among dongs above a fixed 2023 volume floor",
               "volume_floor_daily_mobility": round(floor, 2),
               "volume_floor_source": "2023 stable-dong 60th percentile of 4-week living mobility, fixed before validation/test",
               "forecast_horizon": "13 weeks; trailing four target weeks have all been predicted by the decision origin",
               "validation_weeks": int(val_rank.week.nunique()),
               "test_weeks": int(test_rank.week.nunique()),
               "validation_hits": val_scores.to_dict(),
               "test_hits": test_scores.to_dict(),
               "selected_on_validation": chosen,
               "selected_test_precision_at_20": round(float(test_scores[chosen] / (K * test_rank.week.nunique())), 3),
               "test_note": "Jan-Jul 2025 only; Aug reference affected by 2024 source anomaly",
               "validity_note": "This is a mobility-defined emerging-place proxy, not independently verified cultural hotplace status"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
