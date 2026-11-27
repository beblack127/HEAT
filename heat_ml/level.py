"""Level-change features for v2 step 4 (preprocessing doc A-4, A-5, B-2; catalog F5, F11, P7).

F5   level_ratio = log(mean of the 13 weeks to the origin / mean of the same 13 weeks a year
     earlier); target_lag52_level = target_lag52 + level_ratio (and the festival-aligned version);
     seasonal_level = seasonal × exp(level_ratio) for the raw-scale blend.
F11  Latest step change in the dong's Seoul share as of the origin (share = dong / 426-dong sum of
     the corrected history, festival weeks masked). step(x) at week t = median of x over the next
     ≤13 weeks (at least AFTER_MIN observed) − median over the 13 weeks to t. A week qualifies when
     step(year-on-year log share) ≥ log(1 + STEP_MIN) in size — recurring swings such as the
     semester cancel year on year — and step(log share) moves the same way by ≥ half that, which
     rejects the mirror jump a year after a real change. Needs a year of history (t ≥ SHIFT_FROM). Within the last qualifying run the largest jump is the change point.
     weeks_since_shift (capped at SHIFT_CAP; SHIFT_CAP when none) and shift_size (0 when none).
P7   With drop_pre_shift, training rows of a dong whose origin precedes its change point (as
     detected at the training cutoff) are dropped.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import START, WEEK
from .festival import counterpart
from .kr_calendar import festival_weeks

STEP_MIN = 0.30
AFTER_MIN = 6
SHIFT_CAP = 104
SHIFT_FROM = START + 54 * WEEK
LEVEL_GROUPS = {
    "level": ("level_ratio",),
    "level_target": ("target_lag52_level",),
    "level_target_aligned": ("target_lag52_aligned_level",),
    "shift": ("weeks_since_shift", "shift_size"),
}


def level_ratio(history, origin):
    recent = history.loc[origin - 12 * WEEK:origin]
    earlier = history.loc[origin - 64 * WEEK:origin - 52 * WEEK]
    if len(recent) < 13 or len(earlier) < 13:
        return pd.Series(np.nan, index=history.columns)
    full = recent.notna().sum().eq(13) & earlier.notna().sum().eq(13)
    return np.log(recent.mean() / earlier.mean()).where(full)


def detect_shifts(history, origin):
    """Return (change week or NaT, signed log jump) per dong as of origin."""
    share = history.div(history.sum(axis=1), axis=0)
    normal = pd.Series(~share.index.isin(festival_weeks()), index=share.index)
    log_share = np.log(share.where(normal, axis=0))

    def step(series):
        before = series.rolling(13, min_periods=10).median()
        after = series[::-1].rolling(13, min_periods=AFTER_MIN).median()[::-1].shift(-1)
        return after - before

    jump = step(log_share - log_share.shift(52))   # year-on-year: recurring swings cancel
    level = step(log_share).to_numpy()             # a real change also moves the level itself
    threshold = np.log1p(STEP_MIN)
    values = jump.to_numpy()
    flags = ((np.abs(values) >= threshold) & (level * np.sign(values) >= threshold / 2)
             & np.asarray(jump.index >= SHIFT_FROM)[:, None])
    weeks, sizes = [], []
    for column in range(flags.shape[1]):
        rows = np.flatnonzero(flags[:, column])
        if rows.size == 0:
            weeks.append(pd.NaT)
            sizes.append(0.0)
            continue
        end = rows[-1]
        start = end
        while start - 1 >= 0 and flags[start - 1, column]:
            start -= 1
        best = start + int(np.nanargmax(np.abs(values[start:end + 1, column])))
        weeks.append(jump.index[best])
        sizes.append(float(values[best, column]))
    return pd.DataFrame({"shift_week": weeks, "shift_size": sizes}, index=history.columns)


class LevelMixin:
    """Adds level and step-change columns; optionally drops pre-change training rows (P7)."""

    drop_pre_shift = False

    @lru_cache(maxsize=512)
    def shifts(self, origin):
        return detect_shifts(self.corrected(origin), origin)

    def examples(self, origins, label_cutoff=None):
        frames = []
        for origin in origins:
            frame = super().examples([origin], label_cutoff)
            if frame.empty:
                frames.append(frame)
                continue
            history = self.corrected(origin)
            ratio = level_ratio(history, origin).reindex(frame.unit).to_numpy()
            frame["level_ratio"] = ratio
            frame["target_lag52_level"] = frame.target_lag52 + ratio
            frame["seasonal_level"] = frame.seasonal * np.exp(ratio)
            weeks = {week: counterpart(week) for week in frame.week_end.unique()}
            pairs = pd.MultiIndex.from_arrays([frame.week_end.map(weeks), frame.unit])
            aligned = history.stack(future_stack=True).reindex(pairs).to_numpy()
            frame["target_lag52_aligned_level"] = np.log1p(aligned) + ratio
            frame["seasonal_aligned_level"] = aligned * np.exp(ratio)
            found = self.shifts(origin).reindex(frame.unit)
            since = ((origin - found.shift_week) / WEEK).clip(upper=SHIFT_CAP).fillna(SHIFT_CAP)
            frame["weeks_since_shift"] = since.to_numpy()
            frame["shift_size"] = found.shift_size.to_numpy()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @lru_cache(maxsize=3)
    def training(self, cutoff):
        frame = super().training(cutoff)
        if not self.drop_pre_shift:
            return frame
        change = self.shifts(cutoff).shift_week.reindex(frame.unit).to_numpy()
        keep = pd.isna(change) | (frame.origin.to_numpy() >= change)
        return frame.loc[keep].reset_index(drop=True)
