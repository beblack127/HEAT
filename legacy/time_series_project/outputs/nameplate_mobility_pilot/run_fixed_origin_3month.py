"""Forecast the next 13 complete weeks once, with no observed-value updates."""
from __future__ import annotations

import argparse
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

BASE = Path(__file__).resolve().parent
OUT = BASE / "fixed_origin_3month"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SPECS = {
    "ARIMA": ((0, 1, 1), (0, 0, 0, 0), False),
    "SARIMA": ((1, 1, 0), (1, 0, 0, 13), False),
    "ARIMAX": ((0, 1, 1), (0, 0, 0, 0), True),
    "SARIMAX": ((1, 1, 0), (1, 0, 0, 13), True),
}
ORIGINS = {"backtest": pd.Timestamp("2025-04-21"),
           "final": pd.Timestamp("2025-08-25")}


def fit_forecast(code: str, dong: pd.DataFrame, origin_name: str) -> str:
    origin = ORIGINS[origin_name]
    dong = dong.sort_values("week").set_index("week")
    train = dong.loc[dong.index <= origin].copy()
    if len(train) < 80:
        raise ValueError(f"{code}: too little history")
    future_weeks = pd.date_range(origin + pd.Timedelta(weeks=1), periods=13, freq="W-MON")
    # Every future x is a living-mobility observation from 13 weeks earlier.
    x_future = np.log1p(dong.loc[future_weeks - pd.Timedelta(weeks=13), "y"].to_numpy())
    train["x"] = np.log1p(train.y.shift(13))
    train = train.dropna(subset=["x"])
    for model_name, (order, seasonal, use_x) in SPECS.items():
        file = CACHE / f"{origin_name}_{code}_{model_name}.csv"
        if file.exists():
            continue
        x_mean, x_std = float(train.x.mean()), max(float(train.x.std(ddof=0)), 1e-8)
        x_train = ((train.x - x_mean) / x_std).to_numpy().reshape(-1, 1) if use_x else None
        x_pred = ((x_future - x_mean) / x_std).reshape(-1, 1) if use_x else None
        fit = SARIMAX(np.log1p(train.y.to_numpy()), exog=x_train,
                      order=order, seasonal_order=seasonal, trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit.fit(disp=False, maxiter=200)
        if not result.mle_retvals.get("converged", False):
            with warnings.catch_warnings(record=True) as retry:
                warnings.simplefilter("always")
                result = fit.fit(method="powell", disp=False, maxiter=200)
                caught.extend(retry)
        if not result.mle_retvals.get("converged", False):
            raise RuntimeError(f"{origin_name} {code} {model_name}: fit did not converge")
        pred_log = result.forecast(steps=13, exog=x_pred)
        pred = np.expm1(np.clip(pred_log, 0, 20)).clip(0)
        actual = dong.y.reindex(future_weeks).to_numpy()
        frame = pd.DataFrame({"origin": origin_name, "origin_week": origin.strftime("%Y-%m-%d"),
                              "admin_dong_code": code, "model": model_name,
                              "week": future_weeks.strftime("%Y-%m-%d"),
                              "step": np.arange(1, 14), "actual": actual, "predicted": pred})
        frame.to_csv(file, index=False, encoding="utf-8-sig")
        (CACHE / f"{origin_name}_{code}_{model_name}.json").write_text(json.dumps({
            "train_weeks": len(train), "warnings": len(caught),
            "converged": bool(result.mle_retvals["converged"])}, indent=2), encoding="utf-8")
    return code


def assemble(codes: list[str], origin_name: str, weekly: pd.DataFrame) -> pd.DataFrame:
    files = [CACHE / f"{origin_name}_{code}_{model}.csv" for code in codes for model in SPECS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise ValueError(f"{len(missing)} missing forecast files: {missing[:2]}")
    predictions = pd.concat([pd.read_csv(f, dtype={"admin_dong_code": str}) for f in files], ignore_index=True)
    if len(predictions) != len(codes) * 4 * 13:
        raise ValueError("Unexpected row count")
    # Seasonal naive: known for all 13 future weeks at the fixed origin.
    history = weekly.set_index(["admin_dong_code", "week"])["y"]
    b = predictions.loc[predictions.model.eq("ARIMA")].copy()
    reference = pd.to_datetime(b.week) - pd.Timedelta(weeks=52)
    b["predicted"] = [history.loc[(code, week)] for code, week in zip(b.admin_dong_code, reference)]
    b["model"] = "Prior-year same week"
    predictions = pd.concat([predictions, b], ignore_index=True)
    predictions.to_csv(OUT / f"{origin_name}_predictions.csv", index=False, encoding="utf-8-sig")
    if origin_name == "backtest":
        rows = []
        for model, g in predictions.groupby("model"):
            for scope, q in (("all 13 weeks", g), ("week 13 only", g.loc[g.step.eq(13)])):
                rows.append({"model": model, "scope": scope, "dong_weeks": len(q),
                             "wape_pct": round(float(100 * (q.actual - q.predicted).abs().sum() / q.actual.sum()), 2)})
        pd.DataFrame(rows).to_csv(OUT / "backtest_metrics.csv", index=False)
    return predictions


def plot_fixed_examples(weekly: pd.DataFrame, backtest: pd.DataFrame, final: pd.DataFrame) -> None:
    # Deterministic sample: first code in complete-history set, not selected for accuracy.
    code = sorted(weekly.admin_dong_code.unique())[0]
    names = pd.read_csv(BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
                        "nametags" / "01_행정동_목록.csv", dtype={"dong_code8": str})
    name = names.set_index("dong_code8").dong_name.get(code, "")
    for origin_name, predictions in (("backtest", backtest), ("final", final)):
        origin = ORIGINS[origin_name]
        observed = weekly.loc[weekly.admin_dong_code.eq(code) &
                              weekly.week.between(origin - pd.Timedelta(weeks=20), origin), ["week", "y"]]
        future_actual = weekly.loc[weekly.admin_dong_code.eq(code) &
                                   weekly.week.gt(origin) &
                                   weekly.week.le(origin + pd.Timedelta(weeks=13)), ["week", "y"]]
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True, dpi=180)
        for ax, model in zip(axes.flat, SPECS):
            p = predictions.loc[predictions.admin_dong_code.eq(code) & predictions.model.eq(model)].copy()
            p["week"] = pd.to_datetime(p.week)
            ax.plot(observed.week, observed.y, color="#3368a6", lw=1.8, label="Observed at origin")
            ax.plot(p.week, p.predicted, color="#d3493b", lw=1.8, label="Forecast from fixed origin")
            if len(future_actual):
                ax.plot(future_actual.week, future_actual.y, color="#3368a6", ls="--", lw=1.6,
                        label="Later observed (evaluation)")
            ax.axvline(origin, color="#7a8797", ls="--", lw=1)
            ax.set_title(model, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=25)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{code} {name} | one fixed origin ({origin_name}: {origin:%Y-%m-%d})", fontsize=15)
        fig.text(.07, .006, "Red is one 13-week forecast made at the dashed cutoff; no new observations are fed to the model.",
                 fontsize=9, color="#596675")
        fig.tight_layout(rect=(0, .03, 1, .94))
        fig.savefig(OUT / f"{origin_name}_fixed_origin_example.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    weekly = load_weekly()
    codes = sorted(weekly.admin_dong_code.unique())
    if len(codes) != 424:
        raise ValueError("Expected 424 complete-history dongs")
    output = {}
    for origin_name in ORIGINS:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(fit_forecast, code, weekly.loc[weekly.admin_dong_code.eq(code)].copy(),
                                origin_name): code for code in codes}
            for i, job in enumerate(as_completed(jobs), 1):
                job.result()
                if i % 40 == 0 or i == len(codes):
                    print(f"{origin_name}: {i}/{len(codes)} dongs", flush=True)
        output[origin_name] = assemble(codes, origin_name, weekly)
    plot_fixed_examples(weekly, output["backtest"], output["final"])
    print(pd.read_csv(OUT / "backtest_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
