"""Compare four taught ARIMA-family models on the same mobility panel.

Run from anywhere with Python 3.12 + pandas + statsmodels:
python compare_arima_family.py --repo PATH_TO_Time-series-Project --out OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


SPECS = {
    "ARIMA": ((1, 1, 0), (0, 0, 0, 0), False),
    "SARIMA": ((1, 1, 0), (1, 0, 0, 12), False),
    "ARIMAX": ((1, 1, 0), (0, 0, 0, 0), True),
    "SARIMAX": ((1, 1, 0), (1, 0, 0, 12), True),
}


def load_panel(repo: Path) -> pd.DataFrame:
    folder = repo / "data" / "context_monthly"
    mobility = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                          for p in sorted(folder.glob("mobility_20????.csv"))], ignore_index=True)
    living = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                        for p in sorted(folder.glob("living_20????.csv"))], ignore_index=True)
    if mobility.duplicated(["admin_dong_code", "month"]).any() or living.duplicated(["admin_dong_code", "month"]).any():
        raise ValueError("Duplicate dong-month in source files")
    df = mobility.merge(living[["admin_dong_code", "month", "living_day_mean"]],
                        on=["admin_dong_code", "month"], how="inner", validate="one_to_one")
    df = df.sort_values(["admin_dong_code", "month"]).reset_index(drop=True)
    if len(df) != 40 * 36 or df["admin_dong_code"].nunique() != 40:
        raise ValueError(f"Expected complete 40 x 36 panel, got {df.shape}")
    df["living_lag2"] = df.groupby("admin_dong_code")["living_day_mean"].shift(2)
    df = df[df["month"] >= "2023-03"].copy()
    df["log_y"] = np.log1p(df["mobility_shop_tour_other_daily"])
    df["log_x"] = np.log1p(df["living_lag2"])
    if not np.isfinite(df[["log_y", "log_x"]].to_numpy()).all():
        raise ValueError("Nonfinite target or exogenous values")
    return df


def fit_predict_one(dong: pd.DataFrame, name: str, spec: tuple) -> tuple[list[dict], dict]:
    order, seasonal_order, use_exog = spec
    train = dong[dong["month"] < "2024-12"]
    test = dong[dong["month"] >= "2025-01"]
    if len(train) != 21 or len(test) != 12:
        raise ValueError(f"Unexpected train/test months for {dong['admin_dong_code'].iloc[0]}")
    kwargs = dict(order=order, seasonal_order=seasonal_order, trend="n",
                  enforce_stationarity=True, enforce_invertibility=True)
    exog = train[["log_x"]].to_numpy() if use_exog else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(train["log_y"].to_numpy(), exog=exog, **kwargs)
        result = model.fit(disp=False, maxiter=200)
        optimizer = "lbfgs"
        if not result.mle_retvals.get("converged", False):
            result = model.fit(method="powell", disp=False, maxiter=300)
            optimizer = "powell"
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError(f"Did not converge after two optimizers: {name}, {dong['admin_dong_code'].iloc[0]}")
    output = []
    state_last_month = "2024-11"
    for _, row in test.iterrows():
        future = dong[(dong["month"] > state_last_month) & (dong["month"] <= row["month"])]
        if len(future) != 2:
            raise ValueError(f"Expected exactly two forecast months before {row['month']}")
        next_x = future[["log_x"]].to_numpy() if use_exog else None
        predicted_log = float(result.forecast(steps=2, exog=next_x)[-1])
        predicted = max(0.0, math.expm1(min(predicted_log, 20)))
        output.append({"admin_dong_code": row["admin_dong_code"], "month": row["month"],
                       "model": name, "actual_daily": float(row["mobility_shop_tour_other_daily"]),
                       "predicted_daily": predicted})
        newly_available = future.iloc[0]
        append_x = np.array([[newly_available["log_x"]]]) if use_exog else None
        result = result.append([newly_available["log_y"]], exog=append_x, refit=False)
        state_last_month = newly_available["month"]
    return output, {"admin_dong_code": dong["admin_dong_code"].iloc[0], "model": name,
                    "train_n": len(train), "converged": True, "optimizer": optimizer,
                    "warning_count": len(caught),
                    "warnings": " | ".join(sorted({str(w.message) for w in caught}))}


def score(rows: pd.DataFrame) -> dict:
    error = (rows["actual_daily"] - rows["predicted_daily"]).abs()
    return {"n": int(len(rows)), "mae": round(float(error.mean()), 2),
            "wape": round(float(error.sum() / rows["actual_daily"].sum()), 4),
            "median_dong_mae": round(float(rows.assign(error=error).groupby("admin_dong_code")["error"].mean().median()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dongs", type=int, default=40, help="Debug only; final run uses all 40")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.repo)
    predictions, fitting = [], []
    for i, (code, dong) in enumerate(panel.groupby("admin_dong_code", sort=True)):
        if i >= args.max_dongs:
            break
        for name, spec in SPECS.items():
            rows, fit_info = fit_predict_one(dong, name, spec)
            predictions.extend(rows); fitting.append(fit_info)
        print(f"Completed {i+1}/{min(args.max_dongs, 40)}: {code}", flush=True)
    pred = pd.DataFrame(predictions)
    fit = pd.DataFrame(fitting)
    pred.to_csv(args.out / "arima_family_predictions_2025.csv", index=False, encoding="utf-8-sig")
    fit.to_csv(args.out / "arima_family_fit_audit.csv", index=False, encoding="utf-8-sig")
    report = {"method": "Two-step rolling 2025 forecasts from observations through m-2; coefficients fitted once on 2023-03..2024-11, then newly available target appended without refit.",
              "target": "Daily-average destination movements, purposes 4+5+7",
              "exog": "log1p(daytime living population), 2-month lag; only ARIMAX and SARIMAX",
              "transform": "log1p target and exog; expm1 point forecast",
              "orders": {name: {"order": spec[0], "seasonal_order": spec[1]} for name, spec in SPECS.items()},
              "models": {}, "converged_fits": int(fit["converged"].sum()),
              "warnings": int(fit["warning_count"].sum())}
    for label, sub in (("all_2025", pred), ("jan_aug", pred[pred["month"] < "2025-09"]),
                       ("sep_dec", pred[pred["month"] >= "2025-09"])):
        report["models"][label] = {name: score(g) for name, g in sub.groupby("model")}
    (args.out / "arima_family_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
