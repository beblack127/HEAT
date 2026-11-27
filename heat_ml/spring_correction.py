"""Backtest a day-level March–April 2025 adjustment against the saved Ridge predictions."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (FINAL, MOBILITY_MAP, ORIGINS, ROOT, SEARCH, load_covariates, load_target,
                   read_daily_mobility, sundays)
from .experiment import csv, save_json
from .features import Panel
from .models import forecast, wape
from .phase2 import KEYS, PACKAGE

SPRING_START = pd.Timestamp("2025-03-01")
SPRING_END = pd.Timestamp("2025-04-30")
CONDITION = "only_fourier+search"


def aligned_ratio(observed, reference):
    """Compare means with the observed sample's weekday composition."""
    weekday_means = reference.groupby(reference.index.day_of_week).mean()
    expected = weekday_means.reindex(observed.index.day_of_week)
    if expected.isna().any() or expected.sum() <= 0:
        raise ValueError("Missing weekday-matched city reference.")
    return float(observed.sum() / expected.sum())


class SpringPanel(Panel):
    def __init__(self, target, covariates, daily):
        super().__init__(target, covariates)
        self.city_daily = daily.groupby("date").y.sum()
        self.daily = (daily.assign(unit=daily.unit.replace(MOBILITY_MAP))
                      .pivot_table(index="date", columns="unit", values="y", aggfunc="sum")
                      .reindex(columns=self.raw.columns))
        self.days_in_week = target.pivot(index="week_end", columns="unit", values="n_days")
        self.growth = aligned_ratio(self.city_daily.loc["2024-12"], self.city_daily.loc["2023-12"])

        spring = self.daily.loc[SPRING_START:SPRING_END]
        weeks = sundays(pd.Series(spring.index)).to_numpy()
        by_week = spring.groupby(weeks).mean().iloc[1:-1]
        np.testing.assert_allclose(by_week, self.raw.loc[by_week.index], rtol=1e-9, equal_nan=True)

    @lru_cache(maxsize=256)
    def factors(self, cutoff):
        last = min(cutoff, SPRING_END)
        anomaly_city = self.city_daily.loc[SPRING_START:last]
        monthly = {}
        for month, observed in anomaly_city.groupby(anomaly_city.index.month):
            monthly[int(month)] = aligned_ratio(observed, self.city_daily.loc[f"2024-{month:02}"]) / self.growth

        shares = self.daily.loc["2025-01-06":min(cutoff, pd.Timestamp("2025-06-30"))].div(
            self.city_daily, axis=0).dropna(how="all")
        before = (shares.index < SPRING_START) & ~shares.index.to_series().between("2025-01-27", "2025-02-02").to_numpy()
        after = shares.index > SPRING_END
        normal_shares = shares.loc[before | after].mean()
        abnormal_shares = shares.loc[SPRING_START:last].mean()
        relative = abnormal_shares / normal_shares
        if not np.isfinite(relative).all() or relative.le(0).any():
            raise ValueError(f"Invalid district factors at {cutoff.date()}.")
        return monthly, relative, int(before.sum()), int(after.sum())

    def corrected(self, cutoff):
        values = super().corrected(cutoff)
        if cutoff < SPRING_START:
            return values
        daily = self.daily.loc[SPRING_START:min(cutoff, SPRING_END)]
        monthly, relative, _, _ = self.factors(cutoff)
        factor_per_day = daily.index.to_series().dt.month.map(monthly)
        adjustment = daily.div(factor_per_day, axis=0).div(relative, axis=1) - daily
        weeks = sundays(pd.Series(adjustment.index)).to_numpy()
        delta = adjustment.groupby(weeks).sum(min_count=1).fillna(0)
        values.loc[delta.index] += delta.div(self.days_in_week.loc[delta.index]).where(values.loc[delta.index].notna())
        return values


def evaluate(panel, previous, out, jobs):
    rows, predictions, tuning = [], [], {}
    for origin in ORIGINS:
        print(f"Backtest {origin.date()}", flush=True)
        frame = panel.examples([origin])[[*KEYS, "h", "y"]]
        old = previous.loc[previous.origin.eq(origin)]
        frame = frame.merge(old, on=KEYS, validate="one_to_one")
        if len(frame) != len(old) or not np.allclose(frame.y, frame.reference_y):
            raise ValueError(f"Reference evaluation rows or values changed at {origin.date()}.")
        if origin < SPRING_START:
            frame["spring_corrected"] = frame.baseline
        else:
            pred, audit = forecast(panel, origin, "ridge", CONDITION, jobs, staged=True)
            tuning[str(origin.date())] = audit
            ordered = panel.examples([origin])[KEYS].assign(spring_corrected=pred)
            frame = frame.merge(ordered, on=KEYS, validate="one_to_one")
        # An overlap exists when the Sunday ends at or after March 1 and the Monday begins by April 30.
        frame["spring_overlap"] = frame.week_end.between(SPRING_START, SPRING_END + pd.Timedelta(days=6))
        if not np.isfinite(frame[["baseline", "spring_corrected"]]).all().all():
            raise ValueError(f"Missing predictions at {origin.date()}.")
        predictions.append(frame.drop(columns="reference_y"))
        for scope, block in (("all_weeks", frame), ("exclude_spring_overlap", frame.loc[~frame.spring_overlap])):
            for treatment in ("baseline", "spring_corrected"):
                errors = block.y - block[treatment]
                rows.append({"origin": origin, "scope": scope, "treatment": treatment,
                             "weeks": block.week_end.nunique(), "rows": len(block),
                             "wape": wape(block.y, block[treatment]), "rmse": np.sqrt(errors.pow(2).mean())})
    metrics = pd.DataFrame(rows)
    csv(metrics, out / "outer_metrics.csv")
    summary = metrics.groupby(["scope", "treatment"], as_index=False)[["wape", "rmse"]].mean()
    csv(summary, out / "mean_metrics.csv")
    csv(pd.concat(predictions), out / "outer_predictions.csv")
    save_json(tuning, out / "tuning.json")


def daily_input(out, mobility_dir):
    path = out / "daily_mobility.csv"
    if not path.is_file():
        ranges = (("2023-12-01", "2023-12-31"), ("2024-03-01", "2024-04-30"),
                  ("2024-12-01", "2024-12-31"), ("2025-01-06", "2025-06-30"))
        days = pd.DatetimeIndex([day for start, end in ranges for day in pd.date_range(start, end)])
        temp = path.with_suffix(".tmp")
        csv(read_daily_mobility(mobility_dir, days), temp)
        temp.replace(path)
    return pd.read_csv(path, dtype={"unit": str}, parse_dates=["date"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PACKAGE / "run_spring_correction")
    parser.add_argument("--mobility-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=True)
    daily = daily_input(out, args.mobility_dir)
    target = load_target(PACKAGE / "target_2023_2025.csv")
    inputs = load_covariates({"search": SEARCH}, None, assume_period_end=True)
    panel = SpringPanel(target, inputs, daily)
    name = f"ridge|{CONDITION}"
    previous = pd.read_csv(PACKAGE / "run_phase3/outer_predictions.csv", usecols=[*KEYS, "y", name],
                           parse_dates=["origin", "week_end"], dtype={"unit": str})
    previous = previous.rename(columns={"y": "reference_y", name: "baseline"})
    evaluate(panel, previous, out, args.jobs)

    factor_records = []
    for cutoff in (ORIGINS[2], ORIGINS[3], FINAL):
        monthly, relative, before, after = panel.factors(cutoff)
        factor_records.append({"cutoff": str(cutoff.date()), "city_monthly": {str(k): v for k, v in monthly.items()},
                               "before_days": before, "after_days": after,
                               "district_relative_median": float(relative.median())})
    save_json({"city_growth_december": panel.growth, "factors": factor_records,
               "baseline": "phase3 Ridge with Fourier and search",
               "evaluation": "raw observations; spring interval diagnosed retrospectively"}, out / "setup.json")
    _, relative, _, _ = panel.factors(FINAL)
    csv(relative.rename("relative_factor").reset_index(), out / "district_factors.csv")
    corrected = panel.corrected(FINAL).rename_axis(index="week_end", columns="unit").stack().rename("corrected_y")
    weekly = target.merge(corrected.reset_index(), on=["week_end", "unit"], how="left", validate="one_to_one")
    csv(weekly, out / "weekly_correction_asof_2025-12-28.csv")


if __name__ == "__main__":
    main()
