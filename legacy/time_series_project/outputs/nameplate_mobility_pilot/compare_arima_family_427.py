"""Fixed-order ARIMA-family comparison for the 427-nameplate monthly panel.

Models are fitted per dong where 2023-03..2024-11 and all 2025 months exist.
Forecast for month m uses observations through m-2 and living population m-2.
"""
from __future__ import annotations

import argparse
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


def fit_one(group: pd.DataFrame, name: str, spec: tuple) -> tuple[list[dict], dict]:
    order, seasonal_order, use_exog = spec
    train = group[(group.month >= "2023-03") & (group.month <= "2024-11")].dropna(subset=["log_y", "log_x"])
    test = group[(group.month >= "2025-01") & (group.month <= "2025-12")]
    if len(train) < 19 or len(test) != 12 or test[["log_y", "log_x"]].isna().any().any():
        raise ValueError("insufficient_history_or_target")
    expected_train = pd.period_range(train.month.iloc[0], "2024-11", freq="M").astype(str).tolist()
    if train.month.tolist() != expected_train:
        raise ValueError("noncontiguous_training_history")
    x_train = train[["log_x"]].to_numpy() if use_exog else None
    model = SARIMAX(train.log_y.to_numpy(), exog=x_train, order=order,
                    seasonal_order=seasonal_order, trend="n",
                    enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=200)
        optimizer = "lbfgs"
        if not result.mle_retvals.get("converged", False):
            result = model.fit(method="powell", disp=False, maxiter=300)
            optimizer = "powell"
        if not result.mle_retvals.get("converged", False):
            result = model.fit(method="nm", disp=False, maxiter=300)
            optimizer = "nm"
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("fit_nonconverged")
    state_last_month = "2024-11"
    output = []
    for _, row in test.iterrows():
        future = group[(group.month > state_last_month) & (group.month <= row.month)]
        if len(future) != 2 or future[["log_y", "log_x"]].isna().any().any():
            raise ValueError("missing_state_or_exog")
        x_future = future[["log_x"]].to_numpy() if use_exog else None
        predicted_log = float(result.forecast(steps=2, exog=x_future)[-1])
        if not math.isfinite(predicted_log):
            raise ValueError("nonfinite_forecast")
        predicted = max(0.0, math.expm1(min(predicted_log, 20)))
        output.append({"admin_dong_code": row.admin_dong_code, "gu": row.gu,
                       "dong_name": row.dong_name, "month": row.month,
                       "model": name, "actual_daily": row.target_457_daily,
                       "predicted_daily": predicted, "abs_error": abs(row.target_457_daily - predicted)})
        available = future.iloc[0]
        x_append = np.array([[available.log_x]]) if use_exog else None
        result = result.append([available.log_y], exog=x_append, refit=False)
        state_last_month = available.month
    return output, {"train_n": len(train), "optimizer": optimizer, "fit_warning_count": len(caught),
                    "fit_warnings": " | ".join(sorted({str(w.message) for w in caught}))}


def summarize(group: pd.DataFrame) -> dict:
    return {"n": len(group), "mae": round(float(group.abs_error.mean()), 2),
            "wape": round(float(group.abs_error.sum() / group.actual_daily.sum()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.panel, encoding="utf-8-sig", dtype={"admin_dong_code": str})
    if len(df) != 427 * 36 or df.admin_dong_code.nunique() != 427:
        raise ValueError("Expected 427 x 36 panel")
    df = df.sort_values(["admin_dong_code", "month"]).copy()
    df["living_lag2"] = df.groupby("admin_dong_code").living_day_mean.shift(2)
    df["log_y"] = np.log1p(df.target_457_daily)
    df["log_x"] = np.log1p(df.living_lag2)
    predictions, per_dong = [], []
    for i, (code, group) in enumerate(df.groupby("admin_dong_code", sort=True), 1):
        for name, spec in SPECS.items():
            common = {"admin_dong_code": code, "gu": group.gu.iloc[0],
                      "dong_name": group.dong_name.iloc[0], "model": name}
            try:
                rows, fit = fit_one(group, name, spec)
                predictions.extend(rows)
                sub = pd.DataFrame(rows)
                per_dong.append({**common, "status": "evaluated", "reason": "",
                                 "n_2025": 12, "mae_2025": summarize(sub)["mae"],
                                 "wape_2025": summarize(sub)["wape"],
                                 "mae_jan_aug": summarize(sub[sub.month < "2025-09"])["mae"],
                                 "wape_jan_aug": summarize(sub[sub.month < "2025-09"])["wape"],
                                 **fit})
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                if code in {"11230515", "11230533"} and reason == "insufficient_history_or_target":
                    reason = "created_from_yongsin_2025_08_no_pre_split_series"
                per_dong.append({**common, "status": "not_estimable", "reason": reason,
                                 "train_n": "", "n_2025": 0, "mae_2025": "", "wape_2025": "",
                                 "mae_jan_aug": "", "wape_jan_aug": "",
                                 "optimizer": "", "fit_warning_count": "", "fit_warnings": ""})
        if i % 25 == 0 or i == 427:
            print(f"Completed {i}/427 dongs", flush=True)
            pd.DataFrame(per_dong).to_csv(args.out / "model_metrics_427.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(predictions).to_csv(args.out / "model_predictions_427_2025.csv", index=False, encoding="utf-8-sig")
    pred = pd.DataFrame(predictions)
    scores = pd.DataFrame(per_dong)
    complete = scores.pivot(index="admin_dong_code", columns="model", values="status")
    complete_codes = complete.index[complete.eq("evaluated").all(axis=1)]
    comparable = pred[pred.admin_dong_code.isin(complete_codes)]
    aggregate = {}
    for label, subset in (("all_2025", comparable),
                          ("jan_aug", comparable[comparable.month < "2025-09"]),
                          ("sep_dec", comparable[comparable.month >= "2025-09"])):
        aggregate[label] = {name: summarize(g) for name, g in subset.groupby("model")}
    baseline = df[(df.admin_dong_code.isin(complete_codes)) & (df.month >= "2025-01")].copy()
    all_lags = df.groupby("admin_dong_code").target_457_daily.shift(2)
    baseline["predicted_daily"] = all_lags.loc[baseline.index]
    baseline["actual_daily"] = baseline.target_457_daily
    baseline["abs_error"] = (baseline.actual_daily - baseline.predicted_daily).abs()
    for label, subset in (("all_2025", baseline),
                          ("jan_aug", baseline[baseline.month < "2025-09"]),
                          ("sep_dec", baseline[baseline.month >= "2025-09"])):
        aggregate[label]["lag2_naive"] = summarize(subset)
    result = {"nameplate_dongs": 427, "evaluated_dongs_common": len(complete_codes),
              "not_estimable_dongs": scores[scores.status != "evaluated"][
                  ["admin_dong_code", "dong_name", "reason"]].drop_duplicates().to_dict("records"),
              "method": "Fixed parameters fitted 2023-03..2024-11; 2025 rolling two-step forecast from observations through m-2.",
              "exog": "living_day_mean shifted two months; no Naver data",
              "orders": {name: {"order": spec[0], "seasonal_order": spec[1]} for name, spec in SPECS.items()},
              "aggregate_same_dongs": aggregate}
    (args.out / "model_summary_427.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
