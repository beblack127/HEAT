"""Daily, district-specific correction of the data anomalies (preprocessing doc D-3, P1–P3).

Correction for a window W as of a cutoff, using only days observed by that cutoff:
  F_m  Seoul monthly factor = weekday-matched Seoul mean of month m (holidays excluded)
       / (the same month a year earlier × growth), growth from a normal period before W.
  g_d  district factor = mean daily Seoul share of dong d inside W / mean share on the nearest
       56 reference days before W and the nearest 56 after W (as observed by the cutoff).
       Reference days skip every candidate anomaly window and Seollal/Chuseok weeks; festival
       weeks are also dropped from the inside mean.
  corrected daily value = raw / (F_m × g_d); weekly values are recomputed as Monday–Sunday means.
  *_did variants divide g_d by the same inside/reference ratio on the same days one year earlier.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DONGS, FINAL, MERGE, MOBILITY_MAP, START, read
from .features import Panel
from .kr_calendar import GROWTH, HOLIDAYS, WINDOWS, festival_weeks, week_calendar, week_end, window_days

PACKAGE = Path(__file__).resolve().parent
DAILY = PACKAGE / "daily_2023_2025.csv.gz"
REFERENCE_DAYS = 56
VARIANTS = {
    "legacy": None,                                    # original weekly single Seoul factor
    "daily": ("summer2024", "spring2025"),             # P1 + P2
    "daily_jan": ("summer2024", "jan2025", "spring2025"),  # P1 + P2 + P3(b)
    # g_d divided by the same inside/reference share ratio one year earlier, removing the dong's
    # own seasonality (university dongs: Mar–Apr vs Jan–Feb + May–Jun is ~1.17 in a normal year).
    "daily_did": ("summer2024", "spring2025"),
    "daily_jan_did": ("summer2024", "jan2025", "spring2025"),
    # v3 simple variants: no correction at all, or confirmed anomaly weeks set to missing.
    "none": (),
    "mask": ("summer2024", "spring2025"),
}


def load_daily(path=DAILY):
    """Return (dong × day matrix after HEAT code merges, Seoul daily total before merges)."""
    raw = pd.read_csv(path, dtype={"unit": str}, parse_dates=["date"])
    city = raw.groupby("date").y.sum()
    allowed = read(DONGS).dong_code8.replace(MERGE)
    mapped = raw.assign(unit=raw.unit.replace(MOBILITY_MAP))
    mapped = mapped.loc[mapped.unit.isin(allowed)]
    daily = mapped.pivot_table(index="date", columns="unit", values="y", aggfunc="sum")
    return daily, city


def aligned_ratio(observed, reference):
    """Ratio of means with the reference re-weighted to the observed weekday composition."""
    weekday_means = reference.groupby(reference.index.day_of_week).mean()
    expected = weekday_means.reindex(observed.index.day_of_week)
    if observed.empty or expected.isna().any() or expected.sum() <= 0:
        raise ValueError("Missing weekday-matched Seoul reference.")
    return float(observed.sum() / expected.sum())


class DailyCorrectedPanel(Panel):
    full_window_means = True

    def __init__(self, target, daily, city, windows, year_adjusted=False):
        super().__init__(target, {})
        self.windows = tuple(windows)
        self.year_adjusted = year_adjusted
        self.daily = daily.reindex(columns=self.units).loc[START - pd.Timedelta(days=6):FINAL]
        self.city = city.loc[self.daily.index]
        self.week_of_day = week_end(self.daily.index)
        weekly = self.daily.groupby(self.week_of_day).mean().reindex(self.raw.index[self.raw.index <= FINAL])
        np.testing.assert_allclose(weekly, self.raw.loc[weekly.index], rtol=1e-9, equal_nan=True,
                                   err_msg="Daily mobility does not reproduce the weekly target.")
        self.days_observed = self.daily.notna().groupby(self.week_of_day).sum()
        holidays = self.daily.index.isin(HOLIDAYS)
        self.festival_day = pd.Series(self.week_of_day.isin(festival_weeks()), index=self.daily.index)
        self.normal_city_day = pd.Series(~holidays, index=self.daily.index)
        self.any_window = pd.Series(self.daily.index.isin(window_days(WINDOWS)), index=self.daily.index)

    @lru_cache(maxsize=256)
    def window_factors(self, name, cutoff):
        """Return (monthly Seoul factors, district factors, audit) for one window as of cutoff."""
        start, end = WINDOWS[name]
        last = min(cutoff, end)
        if last < start:
            return None
        city = self.city.loc[self.normal_city_day]
        growth_start, growth_end = GROWTH[name]
        offset = pd.DateOffset(years=1)
        growth = aligned_ratio(city.loc[growth_start:growth_end],
                               city.loc[growth_start - offset:growth_end - offset])
        monthly = {}
        for month, observed in city.loc[start:last].groupby(city.loc[start:last].index.to_period("M")):
            previous = city.loc[str(month - 12)]
            monthly[str(month)] = aligned_ratio(observed, previous) / growth

        shares = self.daily.div(self.city, axis=0)
        observed = shares.index <= cutoff
        eligible = observed & ~self.any_window.to_numpy() & ~self.festival_day.to_numpy()
        before = shares.index[eligible & (shares.index < start)][-REFERENCE_DAYS:]
        after = shares.index[eligible & (shares.index > end)][:REFERENCE_DAYS]
        inside = shares.index[(shares.index >= start) & (shares.index <= last) & ~self.festival_day.to_numpy()]
        relative = shares.loc[inside].mean() / shares.loc[before.append(after)].mean()
        if self.year_adjusted:
            normal = shares.index[~self.any_window.to_numpy() & ~self.festival_day.to_numpy()]
            current = (inside, before.append(after))
            ago = [(days - offset).intersection(normal) for days in current]
            # Prior-year festival weeks are dropped, so allow up to half of the days to be missing.
            if any(len(old) < 0.5 * len(new) for old, new in zip(ago, current)):
                raise ValueError(f"Too few prior-year days to year-adjust {name} at {cutoff.date()}.")
            relative /= shares.loc[ago[0]].mean() / shares.loc[ago[1]].mean()
        if not np.isfinite(relative).all() or relative.le(0).any():
            raise ValueError(f"Invalid district factors for {name} at {cutoff.date()}.")
        audit = {"growth": growth, "monthly": monthly, "before_days": len(before), "after_days": len(after),
                 "inside_days": len(inside), "district_median": float(relative.median()),
                 "district_p05": float(relative.quantile(0.05)), "district_p95": float(relative.quantile(0.95))}
        return monthly, relative, audit

    @lru_cache(maxsize=256)
    def _corrected(self, cutoff):
        daily = self.corrected_daily(cutoff)
        weeks = self.week_of_day[self.daily.index <= cutoff]
        weekly = daily.groupby(weeks).mean().where(self.days_observed.ge(4))
        return weekly.reindex(self.raw.loc[:cutoff].index)

    def corrected(self, cutoff):
        return self._corrected(cutoff).copy()

    def factor(self, cutoff):
        return {name: (None if (f := self.window_factors(name, cutoff)) is None else f[2])
                for name in self.windows}

    def corrected_city(self, cutoff):
        """Seoul daily total divided by the Seoul monthly factor only (for diagnostics)."""
        city = self.city.loc[:cutoff].copy()
        for name in self.windows:
            factors = self.window_factors(name, cutoff)
            if factors is None:
                continue
            start, end = WINDOWS[name]
            days = city.loc[start:min(cutoff, end)].index
            city.loc[days] /= pd.Series(days.to_period("M").astype(str), index=days).map(factors[0])
        return city

    def corrected_daily(self, cutoff):
        daily = self.daily.loc[:cutoff].copy()
        for name in self.windows:
            factors = self.window_factors(name, cutoff)
            if factors is None:
                continue
            monthly, relative, _ = factors
            start, end = WINDOWS[name]
            days = daily.loc[start:min(cutoff, end)].index
            per_day = pd.Series(days.to_period("M").astype(str), index=days).map(monthly)
            daily.loc[days] = daily.loc[days].div(per_day, axis=0).div(relative, axis=1)
        return daily


class LegacyPanel(Panel):
    """The original weekly single-factor correction with the v2 complete-window means."""
    full_window_means = True


class RawPanel(Panel):
    """v3: no anomaly handling; the history is the raw weekly series."""
    full_window_means = True

    def factor(self, cutoff):
        return None

    def corrected(self, cutoff):
        return self.raw.loc[:cutoff].copy()


class MaskedPanel(RawPanel):
    """v3: every week with at least one day inside a listed anomaly window is set to missing,
    both in the feature history and in the training labels. Truth stays raw."""

    def __init__(self, target, covariates, windows):
        super().__init__(target, covariates)
        days = week_calendar(self.weeks)[[f"{name}_days" for name in windows]].sum(axis=1)
        self.masked = self.weeks[days.gt(0).to_numpy()]

    def corrected(self, cutoff):
        values = super().corrected(cutoff)
        values.loc[values.index.isin(self.masked)] = np.nan
        return values


def make_panel(variant, target, daily=None, city=None):
    if variant == "none":
        return RawPanel(target, {})
    if variant == "mask":
        return MaskedPanel(target, {}, VARIANTS[variant])
    if VARIANTS[variant] is None:
        return LegacyPanel(target, {})
    if daily is None:
        daily, city = load_daily()
    return DailyCorrectedPanel(target, daily, city, VARIANTS[variant], year_adjusted=variant.endswith("_did"))
