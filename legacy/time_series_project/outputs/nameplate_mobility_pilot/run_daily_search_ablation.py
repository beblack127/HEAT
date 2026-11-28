"""Compare fixed pooled daily Ridge models with and without Naver search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
DAILY = BASE / "age2030_daily_baseline"
SEARCH = BASE / "daily_search_40" / "dong_daily.csv"
OUT = BASE / "daily_search_ablation"
OUT.mkdir(parents=True, exist_ok=True)
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-08-31")


def load_panel() -> pd.DataFrame:
    codes = set(pd.read_csv(BASE / "age2030_pilot" / "aligned_panel_40.csv",
                            dtype={"admin_dong_code": str}).admin_dong_code)
    files = sorted(DAILY.glob("daily_457_age2030_20????.csv"))
    target = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                        for p in files], ignore_index=True)
    target = target[target.admin_dong_code.isin(codes)]
    target["date"] = pd.to_datetime(target.date.astype(str), format="%Y%m%d")
    search = pd.read_csv(SEARCH, encoding="utf-8-sig", dtype={"admin_dong_code": str})
    search["date"] = pd.to_datetime(search.date)
    panel = target[["admin_dong_code", "date", "age20_39"]].merge(search, on=["admin_dong_code", "date"], validate="one_to_one")
    if len(panel) != 40 * 974 or panel[["age20_39", "search_index"]].isna().any().any():
        raise ValueError("Incomplete 40-dong daily panel")
    return panel.sort_values(["admin_dong_code", "date"]).reset_index(drop=True)


def make_features(panel: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = panel.copy()
    gy = df.groupby("admin_dong_code").age20_39
    gs = df.groupby("admin_dong_code").search_index
    origin_lag = max(horizon, 1)
    df["y_origin"] = np.log1p(gy.shift(origin_lag))
    df["y_prevweek"] = np.log1p(gy.shift(origin_lag + 7))
    df["y_year"] = np.log1p(gy.shift(364))
    df["y_recent7"] = np.log1p(sum(gy.shift(origin_lag + j) for j in range(7)) / 7)
    df["search_origin"] = np.log1p(gs.shift(horizon))
    df["search_recent7"] = np.log1p(sum(gs.shift(horizon + j) for j in range(7)) / 7)
    df["search_week_change"] = df.search_origin - np.log1p(gs.shift(horizon + 7))
    baseline_lag = 7 if horizon in (0, 1, 7) else 35 if horizon == 30 else 364
    df["base_log"] = np.log1p(gy.shift(baseline_lag))
    df["dow"] = df.date.dt.day_of_week.astype(str)
    df["month"] = df.date.dt.month.astype(str)
    df["log_y"] = np.log1p(df.age20_39)
    common = ["y_origin", "y_prevweek", "y_year", "y_recent7", "base_log"]
    added = ["search_origin", "search_recent7", "search_week_change"]
    return df, common, added


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], residual: bool) -> np.ndarray:
    categories = ["admin_dong_code", "dow", "month"]
    mean = train[numeric].mean()
    std = train[numeric].std(ddof=0).replace(0, 1)
    train_num = ((train[numeric] - mean) / std).to_numpy(dtype=float)
    test_num = ((test[numeric] - mean) / std).to_numpy(dtype=float)
    train_cat = pd.get_dummies(train[categories], columns=categories, drop_first=True, dtype=float)
    test_cat = pd.get_dummies(test[categories], columns=categories, drop_first=True, dtype=float)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)
    x_train = np.column_stack([np.ones(len(train)), train_num, train_cat.to_numpy()])
    x_test = np.column_stack([np.ones(len(test)), test_num, test_cat.to_numpy()])
    alpha = np.eye(x_train.shape[1]) * 10.0
    alpha[0, 0] = 0.0
    response = (train.log_y - train.base_log if residual else train.log_y).to_numpy()
    coefficients = np.linalg.solve(x_train.T @ x_train + alpha, x_train.T @ response)
    predicted_log = x_test @ coefficients + (test.base_log.to_numpy() if residual else 0)
    return np.expm1(np.clip(predicted_log, 0, 20))


def main() -> None:
    global SEARCH, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--ages-19to39", action="store_true")
    args = parser.parse_args()
    if args.ages_19to39:
        SEARCH = BASE / "daily_search_19to39_40" / "dong_daily.csv"
        OUT = BASE / "daily_search_19to39_ablation"
        OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    results, audit = [], []
    for horizon in (0, 1, 7, 30, 91):
        df, common, search = make_features(panel, horizon)
        df = df.dropna(subset=common + search + ["log_y"])
        # All fitted coefficients, including preprocessing, are learned before
        # the earliest 2025 forecast origin for this horizon.
        first_origin = TEST_START - pd.Timedelta(days=horizon)
        train = df.loc[df.date.ge("2024-01-01") & df.date.lt(first_origin)]
        test = df.loc[df.date.ge(TEST_START) & df.date.le(TEST_END)]
        if len(test) != 40 * 243 or len(train) < 40 * 200:
            raise ValueError(f"Incomplete train/test at horizon {horizon}")
        for residual in (False, True):
            for label, numeric in (("without_search", common), ("with_search", common + search)):
                model_label = ("naive_residual_" if residual else "direct_") + label
                predicted = fit_predict(train, test, numeric, residual)
                part = test[["admin_dong_code", "date", "age20_39"]].copy()
                part["horizon_days"] = horizon
                part["model"] = model_label
                part["predicted"] = predicted
                part["abs_error"] = (part.age20_39 - part.predicted).abs()
                results.append(part)
                audit.append({"horizon_days": horizon, "model": model_label,
                              "train_start": train.date.min().strftime("%Y-%m-%d"),
                              "train_end": train.date.max().strftime("%Y-%m-%d"),
                              "train_rows": len(train), "test_rows": len(test),
                              "mae": round(float(part.abs_error.mean()), 2),
                              "wape_pct": round(float(100 * part.abs_error.sum() / part.age20_39.sum()), 2)})
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "design.json").write_text(json.dumps({
        "target": "daily 20s+30s purpose 4+5+7 destination mobility",
        "search": ("NAVER age buckets 19-39 (includes age 19), daily HEAT-mapped relative search index, 40 pilot dongs"
                   if args.ages_19to39 else "all-age daily HEAT-mapped relative search index, 40 pilot dongs"),
        "model": "pooled Ridge alpha=10 on log1p target or naive residual; 40 dong fixed effects, calendar, target lags; search ablation adds 3 lagged search features",
        "horizon_zero": "same-day search nowcast diagnostic, not advance forecast",
        "time_split": "train 2024-01-01 through day before earliest 2025 forecast origin; test 2025-01-01 to 2025-08-31",
        "tuning": "fixed alpha; no validation tuning",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(audit).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
