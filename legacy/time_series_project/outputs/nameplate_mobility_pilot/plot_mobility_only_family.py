"""Whole collected 4+5+7 series, diagnostics, and mobility-only ARIMA family.

The four forecasts are for the sum of 424 complete-history dongs. This is a
citywide illustration; it must not be interpreted as 424 dong-level forecasts.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE = Path(__file__).resolve().parent
RAW = BASE / "age2030_daily_baseline"
OUT = BASE / "mobility_only_family"
OUT.mkdir(exist_ok=True)
H = 13


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(RAW.glob("daily_457_age2030_20????.csv"))
    daily = pd.concat([pd.read_csv(f, dtype={"admin_dong_code": str}) for f in files], ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily = daily.sort_values(["admin_dong_code", "date"])
    if len(files) != 32 or daily.date.nunique() != 974:
        raise ValueError("Unexpected daily-source coverage")
    all_daily = daily.groupby("date", as_index=False).agg(
        age20=("age20", "sum"), age30=("age30", "sum"),
        age20_39=("age20_39", "sum"), all_ages_457=("all_ages_457", "sum"),
        codes=("admin_dong_code", "nunique"))
    daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
    by_dong = daily.groupby(["admin_dong_code", "week"], as_index=False).agg(
        y=("age20_39", "mean"), days=("date", "nunique"))
    by_dong = by_dong.loc[by_dong.days.eq(7)]
    stable = by_dong.groupby("admin_dong_code").week.nunique()
    stable = stable.index[stable.eq(139)]
    panel = by_dong.loc[by_dong.admin_dong_code.isin(stable)].copy()
    if len(stable) != 424 or len(panel) != 424 * 139:
        raise ValueError("Unexpected complete-history panel")
    weekly = panel.groupby("week", as_index=False).agg(y=("y", "sum"))
    weekly["lag13"] = weekly.y.shift(H)
    weekly["log_y"] = np.log1p(weekly.y)
    weekly["log_lag13"] = np.log1p(weekly.lag13)
    return all_daily, panel, weekly


def plot_history(all_daily: pd.DataFrame, panel: pd.DataFrame, weekly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, dpi=180,
                             gridspec_kw={"height_ratios": [2.1, 1.3, 1.4]})
    for ax in axes:
        ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-31"), color="#f7cfc6", alpha=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].plot(all_daily.date, all_daily.all_ages_457 / 1e6, color="#536479", lw=.8, label="All ages")
    axes[0].plot(all_daily.date, all_daily.age20_39 / 1e6, color="#c64d3b", lw=.9, label="20s + 30s (GT)")
    axes[0].set_ylabel("Daily estimated moves, millions")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("All collected daily 4+5+7 destination mobility | Seoul", loc="left", fontsize=16)
    axes[1].plot(all_daily.date, all_daily.age20 / 1e6, color="#d18135", lw=.9, label="20s")
    axes[1].plot(all_daily.date, all_daily.age30 / 1e6, color="#496eaf", lw=.9, label="30s")
    axes[1].set_ylabel("Daily estimated moves, millions")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(weekly.week, weekly.y / 1e6, color="#c64d3b", lw=1.5)
    axes[2].set_ylabel("Weekly mean daily moves, millions")
    axes[2].set_xlabel("Week starting Monday")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].text(.01, .96, "424 complete-history dongs; 139 complete weeks", transform=axes[2].transAxes,
                 va="top", fontsize=9, color="#536479")
    fig.text(.08, .008, "Daily panels include every observed code (429 over the period). Shaded: unexplained 2024 Aug–Oct mobility surge.\n"
             "Estimated movements are not unique visitors; missing dong-days are not filled with zero.",
             fontsize=8.5, color="#596675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "01_all_collected_457.png", bbox_inches="tight")
    plt.close(fig)

    heat = panel.pivot(index="admin_dong_code", columns="week", values="y")
    heat = heat.div(heat.median(axis=1), axis=0).clip(0, 2.5)
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    image = ax.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest", vmin=.5, vmax=1.8,
                      cmap="RdBu_r")
    tick_ix = np.arange(0, heat.shape[1], 13)
    ax.set_xticks(tick_ix, [pd.Timestamp(heat.columns[i]).strftime("%Y-%m") for i in tick_ix], rotation=45)
    ax.set_yticks([])
    ax.set_ylabel("424 neighborhoods, sorted by code")
    ax.set_title("Every complete-history neighborhood | 4+5+7 mobility relative to its own median", loc="left")
    fig.colorbar(image, ax=ax, label="Weekly level / neighborhood median", shrink=.8)
    fig.tight_layout()
    fig.savefig(OUT / "02_all_424_dong_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(weekly: pd.DataFrame) -> None:
    level = np.log1p(weekly.y)
    diff = level.diff().dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=180)
    plot_acf(level, lags=60, zero=False, ax=axes[0, 0], title="ACF | log mobility level")
    plot_pacf(level, lags=26, zero=False, method="ywm", ax=axes[0, 1], title="PACF | log mobility level")
    plot_acf(diff, lags=60, zero=False, ax=axes[1, 0], title="ACF | first difference of log mobility")
    plot_pacf(diff, lags=26, zero=False, method="ywm", ax=axes[1, 1], title="PACF | first difference of log mobility")
    for ax in axes.flat:
        ax.set_xlabel("Lag (weeks)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Citywide 20s–30s mobility diagnostics | 424 dongs, 139 weeks", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "03_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)


def forecast_family(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_target = pd.Timestamp("2025-01-06")
    last_target = pd.Timestamp("2025-08-25")
    first_origin = first_target - pd.Timedelta(weeks=H)
    specs = {
        "ARIMA": ((0, 1, 1), (0, 0, 0, 0), False),
        "SARIMA": ((1, 1, 0), (1, 0, 0, 13), False),
        "ARIMAX": ((0, 1, 1), (0, 0, 0, 0), True),
        "SARIMAX": ((1, 1, 0), (1, 0, 0, 13), True),
    }
    target_rows = weekly.loc[weekly.week.between(first_target, last_target)]
    outputs = []
    fits = []
    for name, (order, seasonal, use_x) in specs.items():
        # Keep an identical training span for all four models.
        frame = weekly.dropna(subset=["log_lag13"]).copy()
        train = frame.loc[frame.week.le(first_origin)]
        x_mean = float(train.log_lag13.mean()) if use_x else 0.0
        x_std = max(float(train.log_lag13.std(ddof=0)), 1e-8) if use_x else 1.0
        if use_x:
            frame["x"] = (frame.log_lag13 - x_mean) / x_std
            train = frame.loc[frame.week.le(first_origin)]
        model = SARIMAX(train.log_y.to_numpy(),
                        exog=train[["x"]].to_numpy() if use_x else None,
                        order=order, seasonal_order=seasonal, trend="n",
                        enforce_stationarity=True, enforce_invertibility=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(disp=False, maxiter=200)
        if not result.mle_retvals.get("converged", False):
            with warnings.catch_warnings(record=True) as retry:
                warnings.simplefilter("always")
                result = model.fit(method="powell", disp=False, maxiter=200)
                caught.extend(retry)
        if not result.mle_retvals.get("converged", False):
            raise RuntimeError(f"{name}: fit did not converge")
        fits.append({"model": name, "order": str(order), "seasonal_order": str(seasonal),
                     "train_weeks": len(train), "fit_warnings": len(caught)})
        state_last = first_origin
        for row in target_rows.itertuples():
            future = frame.loc[frame.week.gt(state_last) & frame.week.le(row.week)]
            if len(future) != H:
                raise ValueError("Forecast window is not 13 consecutive weeks")
            # Each x_t = mobility at t-13, so all 13 future x values are known at origin.
            pred_log = float(result.forecast(H, exog=future[["x"]].to_numpy() if use_x else None)[-1])
            pred = max(0.0, float(np.expm1(np.clip(pred_log, 0, 20))))
            outputs.append({"week": row.week, "model": name, "actual": row.y, "predicted": pred})
            available = future.iloc[0]
            result = result.append([available.log_y],
                                   exog=np.array([[available.x]]) if use_x else None, refit=False)
            state_last = available.week
    predictions = pd.DataFrame(outputs)
    predictions.to_csv(OUT / "citywide_predictions.csv", index=False)
    metrics = []
    for (model, scope), g in [( (model, scope), group) for (model, scope), group in
                              predictions.assign(scope=lambda p: np.where(p.week.lt("2025-08-01"),
                                                                          "Jan-Jul", "Aug"))
                              .groupby(["model", "scope"])]:
        metrics.append({"model": model, "scope": scope, "weeks": len(g),
                        "wape_pct": round(float(100 * (g.actual - g.predicted).abs().sum() / g.actual.sum()), 2),
                        "mae": round(float((g.actual - g.predicted).abs().mean()), 0)})
    for model, g in predictions.groupby("model"):
        metrics.append({"model": model, "scope": "Jan-Aug", "weeks": len(g),
                        "wape_pct": round(float(100 * (g.actual - g.predicted).abs().sum() / g.actual.sum()), 2),
                        "mae": round(float((g.actual - g.predicted).abs().mean()), 0)})
    metric_frame = pd.DataFrame(metrics).sort_values(["scope", "model"])
    metric_frame.to_csv(OUT / "citywide_metrics.csv", index=False)
    (OUT / "fit_audit.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    return predictions, metric_frame


def plot_forecasts(weekly: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    labels = ["ARIMA", "SARIMA", "ARIMAX", "SARIMAX"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True, dpi=180)
    context = weekly.loc[weekly.week.ge("2024-10-07")]
    for ax, label in zip(axes.flat, labels):
        p = predictions.loc[predictions.model.eq(label)]
        score = metrics.loc[metrics.model.eq(label) & metrics.scope.eq("Jan-Jul"), "wape_pct"].iloc[0]
        ax.plot(context.week, context.y / 1e6, color="#3368a6", lw=1.9, label="Observed mobility")
        ax.plot(p.week, p.predicted / 1e6, color="#d3493b", lw=1.9,
                label=f"{label} prediction")
        ax.axvline(pd.Timestamp("2025-01-06"), color="#7a8797", ls="--", lw=1)
        ax.set_title(f"{label} | Jan–Jul WAPE {score:.2f}%", loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].set_ylabel("Estimated moves per day, millions")
    axes[1, 0].set_ylabel("Estimated moves per day, millions")
    fig.suptitle("Mobility-only ARIMA family | citywide sum, 13-week-ahead rolling forecast", fontsize=16)
    fig.text(.07, .005, "Red begins at the first target week (2025-01-06); each red point is forecast 13 weeks ahead.\n"
             "ARIMAX/SARIMAX X = same mobility series lagged 13 weeks. Citywide aggregate, not dong-level performance.",
             fontsize=8.5, color="#596675")
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(OUT / "04_four_model_forecasts.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_daily, panel, weekly = load()
    all_daily.to_csv(OUT / "all_collected_daily_citywide.csv", index=False)
    weekly.to_csv(OUT / "complete_history_424_weekly_citywide.csv", index=False)
    plot_history(all_daily, panel, weekly)
    plot_diagnostics(weekly)
    predictions, metrics = forecast_family(weekly)
    plot_forecasts(weekly, predictions, metrics)
    weekly["lag52"] = weekly.y.shift(52)
    baseline = []
    for scope, start, end in (("Jan-Jul", "2025-01-06", "2025-07-28"),
                              ("Aug", "2025-08-04", "2025-08-25"),
                              ("Jan-Aug", "2025-01-06", "2025-08-25")):
        part = weekly.loc[weekly.week.between(start, end)]
        baseline.append({"model": "Prior-year same week", "scope": scope, "weeks": len(part),
                         "wape_pct": round(float(100 * (part.y - part.lag52).abs().sum() / part.y.sum()), 2),
                         "mae": round(float((part.y - part.lag52).abs().mean()), 0)})
    pd.concat([metrics, pd.DataFrame(baseline)], ignore_index=True).to_csv(
        OUT / "citywide_metrics_with_baseline.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
