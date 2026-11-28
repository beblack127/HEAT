"""Preliminary mobility forecast ablation with HEAT's mapped Naver search index.

Run from the shared workspace:
  work/.venv-models/Scripts/python.exe outputs/nameplate_mobility_pilot/search_mobility_pilot.py

The 2025 Jan-Aug evaluation is exploratory because this period has already been
examined in earlier mobility work. No hyperparameter selection is performed.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MOBILITY_REPO = ROOT / "work" / "Time-series-Project"
SEARCH_REPO = ROOT / "work" / "HEAT"
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "search_exog_pilot"
TRAIN_START = "2023-03"
TRAIN_END = "2024-11"
TEST_START = "2025-01"
TEST_END = "2025-08"
SPECS = {
    "ARIMA": (),
    "ARIMAX_search": ("search_x",),
    "ARIMAX_living": ("living_x",),
    "ARIMAX_both": ("search_x", "living_x"),
    "SARIMA": (),
    "SARIMAX_search": ("search_x",),
    "SARIMAX_living": ("living_x",),
    "SARIMAX_both": ("search_x", "living_x"),
}


def load_panel(search_measure: str) -> tuple[pd.DataFrame, dict]:
    folder = MOBILITY_REPO / "data" / "context_monthly"
    mobility = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
         for p in sorted(folder.glob("mobility_20????.csv"))], ignore_index=True
    )
    living = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
         for p in sorted(folder.glob("living_20????.csv"))], ignore_index=True
    )
    search_path = SEARCH_REPO / "data" / "processed" / "search" / "행정동_검색지수_월별.csv"
    search = pd.read_csv(search_path, encoding="utf-8-sig", dtype={"dong_code8": str})
    search = search.loc[search.tier.eq("전체"), ["dong_code8", "month", "seoul_share", "index"]].rename(
        columns={"dong_code8": "admin_dong_code", "seoul_share": "search_share", "index": "search_index"}
    )
    for name, table in (("mobility", mobility), ("living", living), ("search", search)):
        if table.duplicated(["admin_dong_code", "month"]).any():
            raise ValueError(f"Duplicate dong-month: {name}")
    panel = mobility.merge(
        living[["admin_dong_code", "month", "living_day_mean"]],
        on=["admin_dong_code", "month"], how="left", validate="one_to_one",
    ).merge(search, on=["admin_dong_code", "month"], how="left", validate="one_to_one")
    panel = panel.sort_values(["admin_dong_code", "month"]).reset_index(drop=True)
    panel["living_lag2"] = panel.groupby("admin_dong_code").living_day_mean.shift(2)
    source_col = "search_share" if search_measure == "share" else "search_index"
    panel["search_lag2"] = panel.groupby("admin_dong_code")[source_col].shift(2)
    panel["log_y"] = np.log1p(panel.mobility_shop_tour_other_daily)
    # Search is a Seoul-relative share, not a count. One million scales the
    # small fractions before log1p; the factor is fixed before evaluation.
    panel["search_x"] = np.log1p((1_000_000 if search_measure == "share" else 1) * panel.search_lag2)
    panel["living_x"] = np.log1p(panel.living_lag2)
    panel = panel.loc[panel.month.ge(TRAIN_START) & panel.month.le(TEST_END)].copy()
    coverage = {
        "mobility_dongs": int(mobility.admin_dong_code.nunique()),
        "mobility_months": int(mobility.month.nunique()),
        "joined_rows": int(len(panel)),
        "search_share_nonmissing": int(panel.search_share.notna().sum()),
        "search_lag2_nonmissing": int(panel.search_lag2.notna().sum()),
        "search_lag2_zero": int(panel.search_lag2.eq(0).sum()),
        "living_lag2_nonmissing": int(panel.living_lag2.notna().sum()),
        "search_dongs_with_full_required_period": int(panel.groupby("admin_dong_code").search_x.apply(lambda s: s.notna().all()).sum()),
        "search_source": str(search_path),
        "search_measure": search_measure,
    }
    return panel, coverage


def fit_predict_one(dong: pd.DataFrame, name: str, features: tuple[str, ...]) -> tuple[list[dict], dict]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train = dong.loc[dong.month.le(TRAIN_END)].copy()
    test = dong.loc[dong.month.ge(TEST_START) & dong.month.le(TEST_END)].copy()
    if len(train) != 21 or len(test) != 8:
        raise ValueError("Incomplete train/test months")
    if not np.isfinite(dong[["log_y", *features]].to_numpy()).all():
        raise ValueError("Missing or nonfinite target/features")
    # Exogenous scaling is estimated on the training months of this dong only.
    scaled = dong.copy()
    scales = {}
    for feature in features:
        mean = float(train[feature].mean())
        std = float(train[feature].std(ddof=0))
        std = std if std > 1e-8 else 1.0
        scaled[feature] = (scaled[feature] - mean) / std
        scales[feature] = {"train_mean": mean, "train_std": std}
    train = scaled.loc[scaled.month.le(TRAIN_END)]
    test = scaled.loc[scaled.month.ge(TEST_START) & scaled.month.le(TEST_END)]
    x_train = train[list(features)].to_numpy() if features else None
    model = SARIMAX(
        train.log_y.to_numpy(), exog=x_train,
        order=(1, 1, 0),
        seasonal_order=(1, 0, 0, 12) if name.startswith("SARIM") else (0, 0, 0, 0),
        trend="n",
        enforce_stationarity=True, enforce_invertibility=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=200)
        optimizer = "lbfgs"
        if not result.mle_retvals.get("converged", False):
            result = model.fit(method="powell", disp=False, maxiter=300)
            optimizer = "powell"
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("fit_nonconverged")
    output = []
    state_last_month = TRAIN_END
    for _, row in test.iterrows():
        future = scaled.loc[scaled.month.gt(state_last_month) & scaled.month.le(row.month)]
        if len(future) != 2:
            raise ValueError("Expected two-step window")
        x_future = future[list(features)].to_numpy() if features else None
        predicted_log = float(result.forecast(steps=2, exog=x_future)[-1])
        if not math.isfinite(predicted_log):
            raise ValueError("nonfinite_forecast")
        forecast = max(0.0, math.expm1(min(predicted_log, 20)))
        output.append({
            "admin_dong_code": row.admin_dong_code, "month": row.month,
            "model": name, "actual_daily": float(row.mobility_shop_tour_other_daily),
            "predicted_daily": forecast, "abs_error": abs(float(row.mobility_shop_tour_other_daily) - forecast),
        })
        available = future.iloc[0]
        x_append = np.array([[available[f] for f in features]]) if features else None
        result = result.append([available.log_y], exog=x_append, refit=False)
        state_last_month = available.month
    return output, {"optimizer": optimizer, "warnings": len(caught), "scales": scales}


def score(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "mae": round(float(df.abs_error.mean()), 2),
        "wape_pct": round(100 * float(df.abs_error.sum() / df.actual_daily.sum()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--search-measure", choices=["share", "index"], default="share")
    args = parser.parse_args()
    out = OUT if args.search_measure == "share" else OUT / "index_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    panel, coverage = load_panel(args.search_measure)
    (out / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False), flush=True)
    complete = panel.groupby("admin_dong_code").apply(
        lambda g: len(g) == 30 and np.isfinite(g[["log_y", "search_x", "living_x"]].to_numpy()).all(),
        include_groups=False,
    )
    codes = complete.index[complete].tolist()
    print(f"Complete comparable dongs: {len(codes)}", flush=True)
    if args.audit_only:
        return
    if not codes:
        raise ValueError("No complete dong panel")
    predictions, fit_audit = [], []
    for i, code in enumerate(codes, 1):
        dong = panel.loc[panel.admin_dong_code.eq(code)].copy()
        for name, features in SPECS.items():
            rows, audit = fit_predict_one(dong, name, features)
            predictions.extend(rows)
            fit_audit.append({"admin_dong_code": code, "model": name,
                              "optimizer": audit["optimizer"], "warnings": audit["warnings"]})
        if i % 5 == 0 or i == len(codes):
            print(f"Completed {i}/{len(codes)} dongs", flush=True)
    pred = pd.DataFrame(predictions)
    fits = pd.DataFrame(fit_audit)
    if pred.groupby(["admin_dong_code", "month"]).model.nunique().ne(len(SPECS)).any():
        raise ValueError("Unequal evaluation rows between models")
    pred.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    fits.to_csv(out / "fit_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment": "40-dong exploratory mobility forecast with lagged HEAT Naver search",
        "target": "Daily-average destination mobility, purpose codes 4+5+7",
        "search": ("HEAT tier=전체 seoul_share; two-month lag; fixed transform log1p(1e6*share)"
                   if args.search_measure == "share" else
                   "HEAT tier=전체 common-anchor relative index; two-month lag; log1p transform"),
        "living": "Daytime living population; two-month lag; log1p transform",
        "method": "Per-dong fixed ARIMA(1,1,0) with or without seasonal AR(1) at 12 months; 2-step rolling forecast from m-2; coefficients fitted once, state updated without refit",
        "train": f"{TRAIN_START} to {TRAIN_END}",
        "test": f"{TEST_START} to {TEST_END}",
        "coverage": coverage,
        "comparable_dongs": len(codes),
        "models": {name: score(group) for name, group in pred.groupby("model")},
        "caveat": "Exploratory retrospective evaluation; Jan-Aug 2025 and model order were already examined in earlier work.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["models"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
