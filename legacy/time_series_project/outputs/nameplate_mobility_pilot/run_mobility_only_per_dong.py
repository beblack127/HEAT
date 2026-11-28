"""Fair 13-week mobility-only ARIMA-family comparison, one model per dong."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import run_weekly_arima_family as family
from run_weekly_citywide_transfer import load_weekly

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
OUT = BASE / "mobility_only_dong"
CACHE = OUT / "cache"
MODEL_SPECS = {
    "ARIMA": ((0, 1, 1), (0, 0, 0, 0), ()),
    "SARIMA": ((1, 1, 0), (1, 0, 0, 13), ()),
    "ARIMAX": ((0, 1, 1), (0, 0, 0, 0), ("self_lag13",)),
    "SARIMAX": ((1, 1, 0), (1, 0, 0, 13), ("self_lag13",)),
}
family.SPECS.update(MODEL_SPECS)


def forecast_dong(code: str, dong: pd.DataFrame) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    for label in MODEL_SPECS:
        file = CACHE / f"{code}_{label}.csv"
        if file.exists():
            continue
        predicted, audit = family.forecast_one(dong, "test", label)
        predicted.to_csv(file, index=False, encoding="utf-8-sig")
        (CACHE / f"{code}_{label}.json").write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    return code


def prepare() -> pd.DataFrame:
    weekly = load_weekly()
    weekly = weekly.sort_values(["admin_dong_code", "week"])
    weekly["self_lag13"] = np.log1p(weekly.groupby("admin_dong_code").y.shift(13))
    # Identical learning span for all four models; this is needed by ARIMAX.
    weekly = weekly.dropna(subset=["self_lag13"]).copy()
    if weekly.groupby("admin_dong_code").size().nunique() != 1:
        raise ValueError("Unequal dong history")
    return weekly


def collect(codes: list[str], weekly: pd.DataFrame, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = [CACHE / f"{code}_{model}.csv" for code in codes for model in MODEL_SPECS]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise ValueError(f"Missing {len(missing)} forecasts; first: {missing[:3]}")
    predictions = pd.concat([pd.read_csv(p, dtype={"admin_dong_code": str}) for p in files], ignore_index=True)
    baseline = weekly.loc[weekly.admin_dong_code.isin(codes) &
                          weekly.week.between("2025-01-06", "2025-08-25"),
                          ["admin_dong_code", "week", "y", "lag52"]].copy()
    baseline["week"] = baseline.week.dt.strftime("%Y-%m-%d")
    baseline = baseline.rename(columns={"y": "actual", "lag52": "predicted"})
    baseline["model"] = "Prior-year same week"
    baseline["fold"] = "test"
    predictions = pd.concat([predictions, baseline], ignore_index=True)
    predictions["abs_error"] = (predictions.actual - predictions.predicted).abs()
    predictions["scope"] = np.where(predictions.week.lt("2025-08-01"), "Jan-Jul", "Aug")
    if predictions.duplicated(["admin_dong_code", "week", "model"]).any():
        raise ValueError("Duplicate predictions")
    if len(predictions) != len(codes) * 34 * 5:
        raise ValueError("Unexpected prediction count")
    predictions.to_csv(OUT / f"{suffix}_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = []
    for (scope, model), g in predictions.groupby(["scope", "model"]):
        metrics.append({"scope": scope, "model": model, "dong_weeks": len(g),
                        "wape_pct": round(float(100 * g.abs_error.sum() / g.actual.sum()), 2),
                        "mae": round(float(g.abs_error.mean()), 2)})
    for model, g in predictions.groupby("model"):
        metrics.append({"scope": "Jan-Aug", "model": model, "dong_weeks": len(g),
                        "wape_pct": round(float(100 * g.abs_error.sum() / g.actual.sum()), 2),
                        "mae": round(float(g.abs_error.mean()), 2)})
    metrics = pd.DataFrame(metrics).sort_values(["scope", "model"])
    metrics.to_csv(OUT / f"{suffix}_metrics.csv", index=False)
    per_dong = predictions.loc[predictions.scope.eq("Jan-Jul")].groupby(
        ["admin_dong_code", "model"], as_index=False).agg(error=("abs_error", "sum"), actual=("actual", "sum"))
    per_dong["wape_pct"] = 100 * per_dong.error / per_dong.actual
    per_dong.to_csv(OUT / f"{suffix}_per_dong_metrics.csv", index=False)
    return predictions, metrics


def plot_overview(metrics: pd.DataFrame, suffix: str) -> None:
    data = metrics.loc[metrics.scope.eq("Jan-Jul")].copy().sort_values("wape_pct")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    ax.barh(data.model, data.wape_pct, color=["#3368a6" if x == "Prior-year same week" else "#d3493b"
                                             for x in data.model])
    ax.invert_yaxis()
    for i, value in enumerate(data.wape_pct):
        ax.text(value + .15, i, f"{value:.2f}%", va="center")
    ax.set_xlim(0, data.wape_pct.max() * 1.2)
    ax.set_xlabel("WAPE across dong-week observations (%)")
    ax.set_title(f"Mobility-only per-dong forecasts | {suffix}, Jan–Jul 2025", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{suffix}_wape_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_dong_pages(predictions: pd.DataFrame, codes: list[str], suffix: str,
                    weekly: pd.DataFrame) -> None:
    names = pd.read_csv(BASE.parents[1] / "work" / "HEAT" / "data" / "processed" /
                        "nametags" / "01_행정동_목록.csv", dtype={"dong_code8": str})
    name_map = names.drop_duplicates("dong_code8").set_index("dong_code8").dong_name.to_dict()
    # A multipage PDF keeps all dong-level plots reviewable without 424 loose PNGs.
    with PdfPages(OUT / f"{suffix}_dong_forecasts.pdf") as pdf:
        for code in codes:
            dong = predictions.loc[predictions.admin_dong_code.eq(code)].copy()
            dong["week"] = pd.to_datetime(dong.week)
            actual = weekly.loc[weekly.admin_dong_code.eq(code) &
                                weekly.week.ge("2024-10-07"), ["week", "y"]]
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
            for ax, model in zip(axes.flat, MODEL_SPECS):
                part = dong.loc[dong.model.eq(model)]
                wape = 100 * (part.actual - part.predicted).abs().sum() / part.actual.sum()
                ax.plot(actual.week, actual.y, color="#3368a6", lw=1.6, label="Observed")
                ax.plot(part.week, part.predicted, color="#d3493b", lw=1.5, label="13-week forecast")
                ax.axvline(pd.Timestamp("2025-01-06"), color="#7a8797", ls="--", lw=.8)
                ax.set_title(f"{model} | WAPE {wape:.1f}%", loc="left")
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.2)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{code} {name_map.get(code, '')} | 20s–30s purpose 4+5+7 mobility", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, .96))
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot40", "all424"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    weekly = prepare()
    codes = sorted(weekly.admin_dong_code.unique())
    if args.scope == "pilot40":
        pilot = pd.read_csv(BASE / "weekly_40" / "weekly_panel_40.csv", dtype={"admin_dong_code": str})
        codes = sorted(set(pilot.admin_dong_code))
        if len(codes) != 40:
            raise ValueError("Pilot must have 40 dongs")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(forecast_dong, code, weekly.loc[weekly.admin_dong_code.eq(code)].copy()): code
                 for code in codes}
        for i, future in enumerate(as_completed(tasks), 1):
            code = future.result()
            if i % 20 == 0 or i == len(codes):
                print(f"{args.scope}: {i}/{len(codes)} dongs completed (last {code})", flush=True)
    predictions, metrics = collect(codes, weekly, args.scope)
    plot_overview(metrics, args.scope)
    plot_dong_pages(predictions, codes, args.scope, weekly)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
