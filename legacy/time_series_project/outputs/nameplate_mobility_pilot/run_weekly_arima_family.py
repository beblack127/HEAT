"""13-week rolling weekly ARIMA-family comparison on 40 pilot dongs."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "weekly_40" / "arima_family"
OUT.mkdir(parents=True, exist_ok=True)
HORIZON = 13
FOLDS = {"validation": ("2024-05-06", "2024-07-29"),
         "test": ("2025-01-06", "2025-08-25")}
SPECS = {
    "ARIMA_110": ((1, 1, 0), (0, 0, 0, 0), ()),
    "ARIMA_011": ((0, 1, 1), (0, 0, 0, 0), ()),
    "SARIMA_annual52": ((1, 1, 0), (1, 0, 0, 52), ()),
    "SARIMA_quarter13": ((1, 1, 0), (1, 0, 0, 13), ()),
    "ARIMAX_search_age": ((1, 1, 0), (0, 0, 0, 0), ("search_age_lag13",)),
    "ARIMAX_subway": ((1, 1, 0), (0, 0, 0, 0), ("subway_lag13",)),
    "ARIMAX_both": ((1, 1, 0), (0, 0, 0, 0), ("search_age_lag13", "subway_lag13")),
    "ARIMAX_011_search_age": ((0, 1, 1), (0, 0, 0, 0), ("search_age_lag13",)),
    "ARIMAX_011_subway": ((0, 1, 1), (0, 0, 0, 0), ("subway_lag13",)),
    "ARIMAX_011_both": ((0, 1, 1), (0, 0, 0, 0), ("search_age_lag13", "subway_lag13")),
    "ARIMAX_011_living": ((0, 1, 1), (0, 0, 0, 0), ("living_last_complete_month",)),
    "ARIMAX_011_all3": ((0, 1, 1), (0, 0, 0, 0), ("search_age_lag13", "subway_lag13", "living_last_complete_month")),
    "SARIMAX_quarter13_both": ((1, 1, 0), (1, 0, 0, 13), ("search_age_lag13", "subway_lag13")),
}


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(BASE / "weekly_40" / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
    df["week"] = pd.to_datetime(df.week)
    df = df.sort_values(["admin_dong_code", "week"]).reset_index(drop=True)
    for name, col in (("search_age_lag13", "search_age19_39"), ("subway_lag13", "subway_exits")):
        df[name] = np.log1p(df.groupby("admin_dong_code")[col].shift(HORIZON))
    source_month = ((df.week - pd.Timedelta(weeks=HORIZON)).dt.to_period("M") - 1).astype(str)
    df["living_source_month"] = source_month
    folder = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly"
    living = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                        for p in sorted(folder.glob("living_20????.csv"))], ignore_index=True)
    living = living[["admin_dong_code", "month", "living_day_mean"]]
    df = df.merge(living, left_on=["admin_dong_code", "living_source_month"],
                  right_on=["admin_dong_code", "month"], how="left", validate="many_to_one")
    df["living_last_complete_month"] = np.log1p(df.living_day_mean)
    df["log_y"] = np.log1p(df.y)
    return df


def baseline_predictions(df: pd.DataFrame, fold: str) -> pd.DataFrame:
    start, end = FOLDS[fold]
    frame = df.copy()
    frame["lag13"] = frame.groupby("admin_dong_code").y.shift(13)
    frame["lag52"] = frame.groupby("admin_dong_code").y.shift(52)
    test = frame.loc[frame.week.between(start, end)]
    rows = []
    for label in ("lag13", "lag52"):
        part = test[["admin_dong_code", "week", "y", label]].rename(columns={"y": "actual", label: "predicted"}).copy()
        part["week"] = part.week.dt.strftime("%Y-%m-%d")
        part["model"] = label
        part["fold"] = fold
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def forecast_one(dong: pd.DataFrame, fold: str, label: str) -> tuple[pd.DataFrame, dict]:
    order, seasonal, features = SPECS[label]
    start, end = FOLDS[fold]
    first_target = pd.Timestamp(start)
    first_origin = first_target - pd.Timedelta(weeks=HORIZON)
    train = dong.loc[dong.week.le(first_origin)].copy()
    test = dong.loc[dong.week.between(start, end)].copy()
    if len(test) != (13 if fold == "validation" else 34):
        raise ValueError("Unexpected fold length")
    scales = {}
    scaled = dong.copy()
    for feature in features:
        t = train[feature].dropna()
        mean, std = float(t.mean()), float(t.std(ddof=0))
        std = max(std, 1e-8)
        scaled[feature] = (scaled[feature] - mean) / std
        scales[feature] = (mean, std)
    if features:
        scaled = scaled.dropna(subset=list(features)).reset_index(drop=True)
    train = scaled.loc[scaled.week.le(first_origin)]
    if len(train) < 40:
        raise ValueError("Insufficient training weeks")
    exog = train[list(features)].to_numpy() if features else None
    model = SARIMAX(train.log_y.to_numpy(), exog=exog, order=order, seasonal_order=seasonal,
                    trend="n", enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=150)
    if not result.mle_retvals.get("converged", False):
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            result = model.fit(method="powell", disp=False, maxiter=200)
            caught.extend(second)
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("Fit did not converge")
    rows = []
    state_last = first_origin
    for row in test.itertuples():
        future = scaled.loc[scaled.week.gt(state_last) & scaled.week.le(row.week)]
        if len(future) != HORIZON:
            raise ValueError("Non-contiguous 13-week forecast window")
        x_future = future[list(features)].to_numpy() if features else None
        forecast_log = float(result.forecast(steps=HORIZON, exog=x_future)[-1])
        prediction = max(0.0, float(np.expm1(np.clip(forecast_log, 0, 20))))
        rows.append({"fold": fold, "model": label, "admin_dong_code": row.admin_dong_code,
                     "week": row.week.strftime("%Y-%m-%d"), "actual": row.y, "predicted": prediction})
        available = future.iloc[0]
        x_append = np.array([[available[f] for f in features]]) if features else None
        result = result.append([available.log_y], exog=x_append, refit=False)
        state_last = available.week
    return pd.DataFrame(rows), {"fold": fold, "model": label, "admin_dong_code": dong.admin_dong_code.iloc[0],
                                "train_weeks": len(train), "warnings": len(caught), "scales": scales}


def run(fold: str, limit: int | None) -> None:
    panel = load_panel()
    codes = sorted(panel.admin_dong_code.unique())
    if limit:
        codes = codes[:limit]
    for i, code in enumerate(codes, 1):
        dong = panel.loc[panel.admin_dong_code.eq(code)].copy()
        for label in SPECS:
            output = OUT / f"{fold}_{label}_{code}.csv"
            if output.exists():
                continue
            predicted, audit = forecast_one(dong, fold, label)
            predicted.to_csv(output, index=False, encoding="utf-8-sig")
            (OUT / f"{fold}_{label}_{code}.json").write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
        if i % 5 == 0 or i == len(codes):
            print(f"{fold}: {i}/{len(codes)} dongs", flush=True)
    if limit:
        return
    prediction_files = [OUT / f"{fold}_{label}_{code}.csv" for code in codes for label in SPECS]
    predicted = pd.concat([pd.read_csv(p, dtype={"admin_dong_code": str})
                           for p in prediction_files], ignore_index=True)
    if len(predicted) != 40 * len(SPECS) * (13 if fold == "validation" else 34):
        raise ValueError("Incomplete model predictions")
    baseline = baseline_predictions(panel, fold)
    predicted = pd.concat([predicted, baseline], ignore_index=True)
    predicted["abs_error"] = (predicted.actual - predicted.predicted).abs()
    predicted.to_csv(OUT / f"{fold}_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = predicted.groupby("model").apply(
        lambda x: pd.Series({"n": len(x), "wape_pct": round(100 * x.abs_error.sum() / x.actual.sum(), 2),
                             "mae": round(x.abs_error.mean(), 2)}), include_groups=False).reset_index()
    metrics.to_csv(OUT / f"{fold}_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.fold, args.limit)
