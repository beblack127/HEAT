"""Add known-at-origin dong Naver search to fixed-origin 13-week forecasts."""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from run_weekly_citywide_transfer import load_weekly
from run_fixed_origin_3month import ORIGINS

BASE = Path(__file__).resolve().parent
OUT = BASE / "fixed_origin_search"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SPECS = {
    "ARIMAX_search": ((0, 1, 1), (0, 0, 0, 0), ["search_x"]),
    "SARIMAX_search": ((1, 1, 0), (1, 0, 0, 13), ["search_x"]),
    "ARIMAX_search+mobility": ((0, 1, 1), (0, 0, 0, 0), ["search_x", "self_lag13"]),
    "SARIMAX_search+mobility": ((1, 1, 0), (1, 0, 0, 13), ["search_x", "self_lag13"]),
}


def source_month(weeks: pd.Series) -> pd.Series:
    return ((weeks - pd.Timedelta(weeks=13)).dt.to_period("M") - 1).astype(str)


def prepare() -> tuple[pd.DataFrame, set[str], pd.DataFrame]:
    weekly = load_weekly().sort_values(["admin_dong_code", "week"])
    search = pd.read_csv(BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
                         "search" / "행정동_검색지수_월별.csv", encoding="utf-8-sig",
                         dtype={"dong_code8": str})
    search = search.loc[search.tier.eq("전체"), ["dong_code8", "month", "index"]].copy()
    if search.duplicated(["dong_code8", "month"]).any():
        raise ValueError("Nonunique monthly search")
    covered = set(weekly.admin_dong_code) & set(search.dong_code8)
    weekly = weekly.loc[weekly.admin_dong_code.isin(covered)].copy()
    weekly["search_month"] = source_month(weekly.week)
    weekly = weekly.merge(search.rename(columns={"dong_code8": "admin_dong_code", "month": "search_month",
                                                 "index": "search_index"}),
                          on=["admin_dong_code", "search_month"], how="left", validate="many_to_one")
    weekly["search_x"] = np.log1p(weekly.search_index)
    weekly["self_lag13"] = np.log1p(weekly.groupby("admin_dong_code").y.shift(13))
    if len(covered) != 419 or weekly.search_x.isna().any() or weekly.search_x.le(0).any():
        raise ValueError("Unexpected search coverage")
    return weekly, covered, search


def forecast_dong(code: str, dong: pd.DataFrame, origin_name: str,
                  future_search: pd.DataFrame) -> str:
    origin = ORIGINS[origin_name]
    dong = dong.sort_values("week")
    train = dong.loc[dong.week.le(origin)].dropna(subset=["self_lag13"]).copy()
    future = future_search.loc[future_search.admin_dong_code.eq(code)].sort_values("week")
    if len(future) != 13 or len(train) < 70:
        raise ValueError(f"{code}: insufficient training/future rows")
    if future.search_month.max() > str(origin.to_period("M") - 1):
        raise ValueError("Future search information leaked")
    for name, (order, seasonal, features) in SPECS.items():
        file = CACHE / f"{origin_name}_{code}_{name}.csv"
        if file.exists():
            continue
        x_train = train[features].to_numpy()
        x_future = future[features].to_numpy()
        mean = x_train.mean(axis=0)
        scale = np.maximum(x_train.std(axis=0), 1e-8)
        x_train = (x_train - mean) / scale
        x_future = (x_future - mean) / scale
        model = SARIMAX(np.log1p(train.y.to_numpy()), exog=x_train,
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
            raise RuntimeError(f"{origin_name} {code} {name} did not converge")
        pred = np.expm1(np.clip(fit.forecast(steps=13, exog=x_future), 0, 20)).clip(0)
        frame = future[["week", "actual", "search_month"]].copy()
        frame.insert(0, "admin_dong_code", code)
        frame.insert(0, "origin", origin_name)
        frame["model"] = name
        frame["step"] = np.arange(1, 14)
        frame["predicted"] = pred
        frame["week"] = frame.week.dt.strftime("%Y-%m-%d")
        frame.to_csv(file, index=False, encoding="utf-8-sig")
        (CACHE / f"{origin_name}_{code}_{name}.json").write_text(json.dumps({
            "train_weeks": len(train), "search_source_month_max": future.search_month.max(),
            "warnings": len(caught), "converged": bool(fit.mle_retvals["converged"])}, indent=2), encoding="utf-8")
    return code


def make_future(weekly: pd.DataFrame, search: pd.DataFrame, origin_name: str) -> pd.DataFrame:
    origin = ORIGINS[origin_name]
    dates = pd.date_range(origin + pd.Timedelta(weeks=1), periods=13, freq="W-MON")
    codes = sorted(weekly.admin_dong_code.unique())
    future = pd.MultiIndex.from_product([codes, dates], names=["admin_dong_code", "week"]).to_frame(index=False)
    future["search_month"] = source_month(future.week)
    lookup = search.rename(columns={"dong_code8": "admin_dong_code", "month": "search_month"}).copy()
    lookup["search_x"] = np.log1p(lookup["index"])
    lookup = lookup[["admin_dong_code", "search_month", "search_x"]]
    future = future.merge(lookup, on=["admin_dong_code", "search_month"], how="left", validate="many_to_one")
    history = weekly.set_index(["admin_dong_code", "week"])["y"]
    source_week = future.week - pd.Timedelta(weeks=13)
    future["self_lag13"] = [np.log1p(history.loc[(code, week)])
                            for code, week in zip(future.admin_dong_code, source_week)]
    future["actual"] = [history.get((code, week), np.nan)
                        for code, week in zip(future.admin_dong_code, future.week)]
    if future[["search_x", "self_lag13"]].isna().any().any():
        raise ValueError("Incomplete known-at-origin features")
    return future


def aggregate(codes: list[str], origin_name: str) -> pd.DataFrame:
    files = [CACHE / f"{origin_name}_{code}_{name}.csv" for code in codes for name in SPECS]
    if any(not f.exists() for f in files):
        raise ValueError("Missing cached predictions")
    result = pd.concat([pd.read_csv(f, dtype={"admin_dong_code": str}) for f in files], ignore_index=True)
    if len(result) != len(codes) * 13 * len(SPECS):
        raise ValueError("Unexpected prediction count")
    result.to_csv(OUT / f"{origin_name}_search_predictions.csv", index=False, encoding="utf-8-sig")
    return result


def evaluate(backtest: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    old = pd.read_csv(BASE / "fixed_origin_3month" / "backtest_predictions.csv",
                      dtype={"admin_dong_code": str})
    old = old.loc[old.admin_dong_code.isin(codes)].copy()
    comparison = pd.concat([old, backtest], ignore_index=True)
    if comparison.duplicated(["admin_dong_code", "week", "model"]).any():
        raise ValueError("Duplicate comparison row")
    rows = []
    for model, g in comparison.groupby("model"):
        for scope, part in (("all 13 weeks", g), ("week 13 only", g.loc[g.step.eq(13)])):
            rows.append({"model": model, "scope": scope, "dong_weeks": len(part),
                         "wape_pct": round(float(100 * (part.actual - part.predicted).abs().sum() / part.actual.sum()), 2)})
    metric = pd.DataFrame(rows)
    metric.to_csv(OUT / "same_419_dong_comparison.csv", index=False)
    return metric


def plot_comparison(metric: pd.DataFrame) -> None:
    data = metric.loc[metric.scope.eq("all 13 weeks")].sort_values("wape_pct")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    color = ["#3368a6" if m == "Prior-year same week" else
             "#d3493b" if "search" in m else "#87929e" for m in data.model]
    ax.barh(data.model, data.wape_pct, color=color)
    ax.invert_yaxis()
    for i, value in enumerate(data.wape_pct):
        ax.text(value + .15, i, f"{value:.2f}%", va="center")
    ax.set_xlim(0, data.wape_pct.max() * 1.18)
    ax.set_xlabel("WAPE over 419 neighborhoods × 13 future weeks (%)")
    ax.set_title("Fixed-origin 3-month forecast | does past Naver search help?", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.16, .007, "Origin: 2025-04-27. Search input comes from completed months known at origin.",
             fontsize=9, color="#596675")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(OUT / "search_419_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_example(weekly: pd.DataFrame, predictions: pd.DataFrame) -> None:
    code = sorted(weekly.admin_dong_code.unique())[0]
    origin = ORIGINS["backtest"]
    obs = weekly.loc[weekly.admin_dong_code.eq(code) &
                     weekly.week.between(origin - pd.Timedelta(weeks=20), origin)]
    future = weekly.loc[weekly.admin_dong_code.eq(code) &
                        weekly.week.gt(origin) & weekly.week.le(origin + pd.Timedelta(weeks=13))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True, dpi=180)
    for ax, name in zip(axes.flat, SPECS):
        p = predictions.loc[predictions.admin_dong_code.eq(code) & predictions.model.eq(name)]
        ax.plot(obs.week, obs.y, color="#3368a6", lw=1.8, label="Observed at origin")
        ax.plot(pd.to_datetime(p.week), p.predicted, color="#d3493b", lw=1.8, label="Forecast with past search")
        ax.plot(future.week, future.y, color="#3368a6", ls="--", lw=1.5, label="Later observed")
        ax.axvline(origin, color="#7a8797", ls="--", lw=1)
        ax.set_title(name, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{code} | fixed-origin 13-week forecast using historical Naver search", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "search_fixed_origin_example.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    weekly, covered, search = prepare()
    codes = sorted(covered)
    results = {}
    for origin_name in ORIGINS:
        future = make_future(weekly, search, origin_name)
        with ProcessPoolExecutor(max_workers=6) as pool:
            jobs = {pool.submit(forecast_dong, code,
                                weekly.loc[weekly.admin_dong_code.eq(code)].copy(),
                                origin_name, future.loc[future.admin_dong_code.eq(code)].copy()): code
                    for code in codes}
            for i, job in enumerate(as_completed(jobs), 1):
                job.result()
                if i % 40 == 0 or i == len(codes):
                    print(f"{origin_name}: {i}/{len(codes)} dongs", flush=True)
        results[origin_name] = aggregate(codes, origin_name)
    metric = evaluate(results["backtest"], covered)
    plot_comparison(metric)
    plot_example(weekly, results["backtest"])
    print(metric.to_string(index=False))


if __name__ == "__main__":
    main()
