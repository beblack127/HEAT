"""Compare 13- and 52-week seasonal assumptions on the same 419-dong backtest.

This is a conditional seasonal-difference AR(1) regression, not statsmodels SARIMA.
It tests whether the search gain survives an annual seasonal reference without
changing the dong cohort, forecast origin, search series, or forecast horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ORIGIN = pd.Timestamp("2025-04-21")  # week beginning; information cutoff 2025-04-27
H = 13


def load_panel(search_csv: Path) -> pd.DataFrame:
    files = sorted((BASE / "age2030_daily_baseline").glob("daily_457_age2030_20????.csv"))
    daily = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str}) for p in files],
        ignore_index=True,
    )
    daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
    daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
    weekly = daily.groupby(["admin_dong_code", "week"], as_index=False).agg(
        y=("age20_39", "mean"), days=("date", "nunique")
    )
    weekly = weekly[weekly.days.eq(7)].sort_values(["admin_dong_code", "week"])
    stable = weekly.groupby("admin_dong_code").week.nunique()
    weekly = weekly[weekly.admin_dong_code.isin(stable[stable.eq(139)].index)].copy()

    search = pd.read_csv(search_csv, encoding="utf-8-sig", dtype={"dong_code8": str})
    search = search[search.tier.eq("전체")][["dong_code8", "month", "index"]]
    weekly["search_month"] = (
        (weekly.week - pd.Timedelta(weeks=13)).dt.to_period("M") - 1
    ).astype(str)
    weekly = weekly.merge(
        search.rename(columns={"dong_code8": "admin_dong_code", "month": "search_month"}),
        on=["admin_dong_code", "search_month"], how="inner", validate="many_to_one",
    )
    weekly["x_search"] = np.log1p(weekly["index"])
    weekly["x_mobility"] = np.log1p(weekly.groupby("admin_dong_code").y.shift(13))
    assert weekly.admin_dong_code.nunique() == 419
    assert weekly.groupby("admin_dong_code").week.nunique().eq(139).all()
    forecast_rows = weekly.week.gt(ORIGIN) & weekly.week.le(ORIGIN + pd.Timedelta(weeks=H))
    assert weekly.loc[forecast_rows, "search_month"].max() <= "2025-03"
    return weekly


def predict_one(g: pd.DataFrame, season: int, feature: str, drift: bool) -> np.ndarray:
    g = g.sort_values("week")
    train = g.week.le(ORIGIN).sum()
    assert len(g) >= train + H and g.week.iloc[train] == ORIGIN + pd.Timedelta(weeks=1)
    log_y = np.log1p(g.y.to_numpy())
    z = log_y[season:train] - log_y[:train-season]
    # Use identical training rows for all feature sets within a seasonal period.
    t = np.arange(season + 14, train)
    columns = [z[t - season - 1]]
    x = {}
    for name in ("x_mobility", "x_search"):
        v = g[name].to_numpy(dtype=float)
        mu = np.nanmean(v[:train]); sd = np.nanstd(v[:train])
        x[name] = (v - mu) / max(sd, 1e-8)
    active = [] if feature == "none" else (["x_mobility"] if feature == "mobility" else ["x_mobility", "x_search"])
    columns += [x[name][t] - x[name][t-season] for name in active]
    if drift:
        columns.append(np.ones(len(t)))
    design = np.column_stack(columns)
    target = log_y[t] - log_y[t-season]
    assert np.isfinite(design).all() and np.isfinite(target).all()
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    if abs(coef[0]) > 0.99:
        phi = float(np.clip(coef[0], -0.99, 0.99))
        remaining = np.linalg.lstsq(design[:, 1:], target - phi * design[:, 0], rcond=None)[0] if design.shape[1] > 1 else np.array([])
        coef = np.r_[phi, remaining]

    forecasts = []
    previous = log_y[train - 1] - log_y[train - 1 - season]
    for i in range(train, train + H):
        values = [previous]
        values += [x[name][i] - x[name][i-season] for name in active]
        if drift:
            values.append(1.0)
        zhat = float(np.dot(coef, values))
        forecasts.append(float(np.expm1(np.clip(log_y[i-season] + zhat, 0, 20))))
        previous = zhat
    return np.asarray(forecasts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Small smoke-test subset")
    args = parser.parse_args()
    panel = load_panel(args.search_csv)
    codes = sorted(panel.admin_dong_code.unique())[:args.limit or None]
    rows = []
    for code in codes:
        g = panel[panel.admin_dong_code.eq(code)]
        actual = g.loc[g.week.gt(ORIGIN)].head(H)
        assert len(actual) == H and actual.week.max() == ORIGIN + pd.Timedelta(weeks=H)
        for season in (13, 52):
            for drift in (False, True):
                for feature in ("none", "mobility", "mobility_search"):
                    forecast = predict_one(g, season, feature, drift)
                    for week, y, yhat in zip(actual.week, actual.y, forecast):
                        rows.append((code, week.date(), season, int(drift), feature, y, yhat))
    out = BASE / "seasonality_control"
    out.mkdir(exist_ok=True)
    pred = pd.DataFrame(rows, columns=["admin_dong_code", "week", "season_weeks", "drift", "features", "actual", "predicted"])
    pred.to_csv(out / ("predictions_smoke.csv" if args.limit else "predictions.csv"), index=False)
    pred["ae"] = (pred.actual - pred.predicted).abs()
    metrics = pred.groupby(["season_weeks", "drift", "features"]).agg(ae=("ae", "sum"), actual=("actual", "sum"))
    metrics["wape_pct"] = 100 * metrics.ae / metrics.actual
    print(metrics[["wape_pct"]].round(3).to_string())
    if not args.limit:
        metrics[["wape_pct"]].to_csv(out / "pooled_wape.csv")
        dong = pred.groupby(["season_weeks", "drift", "features", "admin_dong_code"]).agg(ae=("ae", "sum"), actual=("actual", "sum"))
        dong["wape_pct"] = 100 * dong.ae / dong.actual
        dong[["wape_pct"]].to_csv(out / "per_dong_wape.csv")


if __name__ == "__main__":
    main()
