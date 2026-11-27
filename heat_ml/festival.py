"""Festival features for v2 step 2 (preprocessing doc A-3, B-1, B-1b, B-2; catalog P5, F1–F4).

P5/F4  Festival-aligned prior-year week (see counterpart): festival-relative within ±1 week of a
       Seollal/Chuseok anchor or when week − 52 would hit last year's festival; otherwise week − 52.
       Applied to the origin lag (lag52_aligned), the target-week value (target_lag52_aligned) and
       the raw-scale blend value (seasonal_aligned).
F1/F2  Weekday public holidays in the target week; pre/anchor/post week dummies per festival.
F3     Dong festival sensitivity as of the origin: median over observed festivals of
       log(share in the anchor week / mean share 2–3 weeks either side), share = dong / 426-dong sum
       of the corrected history. Only festivals whose post weeks are observed by the origin count.
"""

import numpy as np
import pandas as pd

from .data import WEEK
from .kr_calendar import FESTIVAL_DAYS, HOLIDAYS, week_end

ALIGN = 1
AROUND = (-3, -2, 2, 3)
FESTIVAL_GROUPS = {
    "holiday": ("holiday_weekdays", "seollal_pre", "seollal_week", "seollal_post",
                "chuseok_pre", "chuseok_week", "chuseok_post"),
    "festival_sens": ("festival_sens", "festival_sens_x_week"),
}
ANCHORS = {kind: pd.DatetimeIndex(week_end(days)) for kind, days in FESTIVAL_DAYS.items()}


def festival_offset(week, limit=ALIGN):
    """Return (kind, offset in weeks, anchor index) for the nearest anchor within ±limit, else None."""
    for kind, anchors in ANCHORS.items():
        offsets = (week - anchors) / WEEK
        nearest = int(np.argmin(np.abs(offsets)))
        if abs(offsets[nearest]) <= limit:
            return kind, int(offsets[nearest]), nearest
    return None


def counterpart(week):
    """Prior-year comparison week.

    Festival-relative (prior anchor + same offset) when the week is within ±ALIGN of a festival
    or when week − 52 would land within ±ALIGN of the prior year's festival; otherwise week − 52.
    Keeping week − 52 elsewhere preserves the school calendar (e.g. March semester start).
    """
    naive = week - 52 * WEEK
    found = festival_offset(week)
    if found is None:
        hit = festival_offset(naive)
        if hit is None:
            return naive
        kind, _, index = hit
        current = index + 1
        if current >= len(ANCHORS[kind]):
            return naive
        return ANCHORS[kind][index] + int((week - ANCHORS[kind][current]) / WEEK) * WEEK
    kind, offset, index = found
    return naive if index == 0 else ANCHORS[kind][index - 1] + offset * WEEK


def calendar_features(weeks):
    weeks = pd.DatetimeIndex(weeks)
    weekday_holidays = HOLIDAYS[HOLIDAYS.weekday < 5]
    frame = pd.DataFrame(index=weeks)
    frame["holiday_weekdays"] = (pd.Series(1, index=week_end(weekday_holidays)).groupby(level=0).sum()
                                 .reindex(weeks, fill_value=0).astype(float))
    for kind in ANCHORS:
        offsets = pd.Series([festival_offset(week) for week in weeks], index=weeks)
        offset = offsets.map(lambda f, kind=kind: f[1] if f is not None and f[0] == kind else np.nan)
        for name, value in (("pre", -1), ("week", 0), ("post", 1)):
            frame[f"{kind}_{name}"] = offset.eq(value).astype(float)
    frame["counterpart"] = [counterpart(week) for week in weeks]
    return frame


def festival_sensitivity(history, origin):
    share = history.div(history.sum(axis=1), axis=0)
    ratios = []
    for anchors in ANCHORS.values():
        for anchor in anchors:
            around = [anchor + k * WEEK for k in AROUND]
            if max(around) > origin or min(around) < share.index[0]:
                continue
            ratio = np.log(share.loc[anchor] / share.loc[around].mean())
            if ratio.notna().any():
                ratios.append(ratio)
    if not ratios:
        return pd.Series(np.nan, index=history.columns)
    return pd.concat(ratios, axis=1).median(axis=1)


class FestivalMixin:
    """Adds the festival columns to Panel.examples for any Panel subclass."""

    def examples(self, origins, label_cutoff=None):
        calendar = getattr(self, "_festival_calendar", None)
        if calendar is None:
            calendar = self._festival_calendar = calendar_features(self.weeks)
        frames = []
        for origin in origins:
            frame = super().examples([origin], label_cutoff)
            if frame.empty:
                frames.append(frame)
                continue
            history = self.corrected(origin)
            info = calendar.reindex(frame.week_end)
            for column in FESTIVAL_GROUPS["holiday"]:
                frame[column] = info[column].to_numpy()
            pairs = pd.MultiIndex.from_arrays([info.counterpart.to_numpy(), frame.unit])
            stacked = history.stack(future_stack=True)
            frame["seasonal_aligned"] = stacked.reindex(pairs).to_numpy()
            frame["target_lag52_aligned"] = np.log1p(frame.seasonal_aligned)
            lag = history.reindex([counterpart(origin)]).iloc[0]
            frame["lag52_aligned"] = np.log1p(lag.reindex(frame.unit).to_numpy())
            sensitivity = festival_sensitivity(history, origin)
            frame["festival_sens"] = sensitivity.reindex(frame.unit).to_numpy()
            frame["festival_sens_x_week"] = frame.festival_sens * (frame.seollal_week + frame.chuseok_week)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def with_mixins(panel, *mixins):
    """Return the same panel re-classed with feature mixins (e.g. FestivalMixin) enabled."""
    if mixins:
        name = "".join(m.__name__.removesuffix("Mixin") for m in mixins) + type(panel).__name__
        panel.__class__ = type(name, (*mixins, type(panel)), {})
    return panel
