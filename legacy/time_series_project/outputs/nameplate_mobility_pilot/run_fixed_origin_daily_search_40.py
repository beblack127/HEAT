"""Pilot sensitivity: 19–39 Naver daily search, lagged 13 weeks, fixed origins."""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from run_fixed_origin_3month import ORIGINS
from run_fixed_origin_search import SPECS

BASE = Path(__file__).resolve().parent
OUT = BASE / "fixed_origin_daily_search_40"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    mobility = pd.read_csv(BASE / "weekly_40" / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
    mobility["week"] = pd.to_datetime(mobility.week)
    mobility = mobility.sort_values(["admin_dong_code", "week"])
    search = pd.read_csv(BASE / "daily_search_19to39_40" / "dong_daily.csv",
                         dtype={"admin_dong_code": str})
    search["date"] = pd.to_datetime(search.date)
    search["week"] = search.date.dt.to_period("W-SUN").dt.start_time
    search = search.groupby(["admin_dong_code", "week"], as_index=False).agg(
        search_index=("search_index", "mean"), days=("date", "nunique"))
    search = search.loc[search.days.eq(7)]
    mobility = mobility[["admin_dong_code", "week", "y"]].merge(
        search[["admin_dong_code", "week", "search_index"]],
        on=["admin_dong_code", "week"], validate="one_to_one")
    g = mobility.groupby("admin_dong_code")
    mobility["search_x"] = np.log1p(g.search_index.shift(13))
    mobility["self_lag13"] = np.log1p(g.y.shift(13))
    if mobility.admin_dong_code.nunique() != 40 or len(mobility) != 40 * 139:
        raise ValueError("Pilot weekly coverage mismatch")
    return mobility


def run_dong(code: str, dong: pd.DataFrame, origin_name: str) -> str:
    origin = ORIGINS[origin_name]
    train = dong.loc[dong.week.le(origin)].dropna(subset=["search_x", "self_lag13"])
    future_weeks = pd.date_range(origin + pd.Timedelta(weeks=1), periods=13, freq="W-MON")
    by_week = dong.set_index("week")
    source_weeks = future_weeks - pd.Timedelta(weeks=13)
    future = pd.DataFrame({"week": future_weeks,
                           "search_x": np.log1p(by_week.loc[source_weeks, "search_index"].to_numpy()),
                           "self_lag13": np.log1p(by_week.loc[source_weeks, "y"].to_numpy()),
                           "y": by_week.y.reindex(future_weeks).to_numpy()})
    if len(future) != 13 or future[["search_x", "self_lag13"]].isna().any().any():
        raise ValueError(f"{code}: missing future inputs")
    if len(train) < 70:
        raise ValueError(f"{code}: insufficient training")
    for name, (order, seasonal, features) in SPECS.items():
        file = CACHE / f"{origin_name}_{code}_{name}.csv"
        if file.exists():
            continue
        x = train[features].to_numpy()
        mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        model = SARIMAX(np.log1p(train.y.to_numpy()), exog=(x - mean) / std,
                        order=order, seasonal_order=seasonal, trend="n",
                        enforce_stationarity=True, enforce_invertibility=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(disp=False, maxiter=200)
        if not fit.mle_retvals.get("converged", False):
            with warnings.catch_warnings(record=True) as retry:
                warnings.simplefilter("always")
                fit = model.fit(method="powell", disp=False, maxiter=200)
                caught.extend(retry)
        if not fit.mle_retvals.get("converged", False):
            raise RuntimeError(f"{code} {name} did not converge")
        future_x = (future[features].to_numpy() - mean) / std
        pred = np.expm1(np.clip(fit.forecast(steps=13, exog=future_x), 0, 20)).clip(0)
        p = pd.DataFrame({"admin_dong_code": code, "origin": origin_name,
                          "week": future.week.dt.strftime("%Y-%m-%d"),
                          "step": np.arange(1, 14), "model": name,
                          "actual": future.y.to_numpy(), "predicted": pred})
        p.to_csv(file, index=False, encoding="utf-8-sig")
        (CACHE / f"{origin_name}_{code}_{name}.json").write_text(json.dumps({
            "train_weeks": len(train), "warnings": len(caught),
            "latest_search_week_used": str(origin.date())}, indent=2), encoding="utf-8")
    return code


def main() -> None:
    panel = load()
    codes = sorted(panel.admin_dong_code.unique())
    results = {}
    for origin_name in ORIGINS:
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(run_dong, code, panel.loc[panel.admin_dong_code.eq(code)].copy(), origin_name)
                    for code in codes]
            for job in as_completed(jobs):
                job.result()
        files = [CACHE / f"{origin_name}_{code}_{name}.csv" for code in codes for name in SPECS]
        results[origin_name] = pd.concat([pd.read_csv(f, dtype={"admin_dong_code": str}) for f in files], ignore_index=True)
        results[origin_name].to_csv(OUT / f"{origin_name}_predictions.csv", index=False)
        print(f"{origin_name}: {len(codes)} dongs complete", flush=True)
    existing = pd.read_csv(BASE / "fixed_origin_3month" / "backtest_predictions.csv",
                           dtype={"admin_dong_code": str})
    existing = existing.loc[existing.admin_dong_code.isin(codes)]
    monthly = pd.read_csv(BASE / "fixed_origin_search" / "backtest_search_predictions.csv",
                          dtype={"admin_dong_code": str})
    monthly = monthly.loc[monthly.admin_dong_code.isin(codes)]
    daily = results["backtest"].copy()
    daily["model"] += "_daily19to39"
    monthly["model"] += "_monthlyAllAge"
    comparison = pd.concat([existing, daily, monthly], ignore_index=True)
    metric = []
    for model, g in comparison.groupby("model"):
        metric.append({"model": model, "dong_weeks": len(g),
                       "wape_pct": round(float(100 * (g.actual - g.predicted).abs().sum() / g.actual.sum()), 2)})
    metric = pd.DataFrame(metric).sort_values("wape_pct")
    metric.to_csv(OUT / "same_40_dong_comparison.csv", index=False)
    print(metric.to_string(index=False))


if __name__ == "__main__":
    main()
