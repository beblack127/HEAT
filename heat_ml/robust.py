"""Robust lags, Seoul common component and anomaly flags for v2 step 5
(preprocessing doc A-6, B-4, B-5; catalog P6, F8, F8b, F9). Everything uses the corrected
history as of the origin; the anomaly flags use only the calendar.

P6   Each dong's log1p history is clipped to median ± CLIP_MADS × MAD (scaled 1.4826) of the 26
     weeks to the origin; rlag0–4 are clipped lags, rmean4/rmean13 the log of the mean of clipped
     values (complete windows only); med4/med13 are log1p of the plain rolling medians.
F8   Seoul = 426-dong sum of the corrected history. seoul_* are its log lags/means and the log
     value 52 weeks before the target week; share_* the same for the dong's log share.
F8b  log_seoul_forecast = log Seoul 52 weeks before the target + Seoul level ratio (13 weeks to
     the origin vs the same weeks a year earlier), falling back to the 13- then 1-week Seoul mean.
     A condition ending in "@share" fits log1p(y) − log_seoul_forecast (the log share) and adds it
     back, so the model predicts the share and the Seoul level comes from the forecast.
F9   anomaly_recent4/13 = share of days in the last 4/13 weeks to the origin inside a candidate
     anomaly window; anomaly_lag52 = the same for the week 52 weeks before the target.
"""

import numpy as np
import pandas as pd

from .data import WEEK
from .kr_calendar import WINDOWS, week_calendar

CLIP_MADS = 5
ROBUST_GROUPS = {
    "robust_lags": ("rlag0", "rlag1", "rlag2", "rlag3", "rlag4", "rmean4", "rmean13"),
    "medians": ("med4", "med13"),
    "seoul": ("seoul_lag0", "seoul_mean4", "seoul_mean13", "seoul_target_lag52"),
    "share": ("share_lag0", "share_mean4", "share_mean13", "share_target_lag52"),
    "anomaly_flags": ("anomaly_recent4", "anomaly_recent13", "anomaly_lag52"),
}
SHARE_OFFSET = "log_seoul_forecast"


def complete_mean(window, width):
    return window.mean().where(window.notna().sum().eq(width)) if len(window) == width else \
        pd.Series(np.nan, index=window.columns)


class RobustMixin:
    """Adds robust lags, Seoul and share components, and anomaly-day flags."""

    def examples(self, origins, label_cutoff=None):
        calendar = getattr(self, "_anomaly_share", None)
        if calendar is None:
            days = week_calendar(self.weeks)[[f"{name}_days" for name in WINDOWS]].sum(axis=1)
            calendar = self._anomaly_share = days / 7
        frames = []
        for origin in origins:
            frame = super().examples([origin], label_cutoff)
            if frame.empty:
                frames.append(frame)
                continue
            history = self.corrected(origin)
            units = frame.unit
            targets = frame.week_end - 52 * WEEK

            log = np.log1p(history)
            recent = log.loc[origin - 25 * WEEK:origin]
            median = recent.median()
            mad = (recent - median).abs().median() * 1.4826
            spread = (CLIP_MADS * mad).where(mad.gt(0), np.inf)
            clipped = log.clip(median - spread, median + spread, axis=1)
            columns = {}
            for lag in range(5):
                columns[f"rlag{lag}"] = clipped.reindex([origin - lag * WEEK]).iloc[0]
            for width in (4, 13):
                window = clipped.loc[origin - (width - 1) * WEEK:origin]
                columns[f"rmean{width}"] = np.log(complete_mean(np.expm1(window), width) + 1)
                plain = history.loc[origin - (width - 1) * WEEK:origin]
                columns[f"med{width}"] = np.log1p(plain.median().where(plain.notna().sum().eq(width)))

            seoul = history.sum(axis=1, min_count=1)
            log_seoul = np.log(seoul)
            share = np.log(history.div(seoul, axis=0))
            for width in (4, 13):
                window = history.loc[origin - (width - 1) * WEEK:origin]
                full = len(window) == width
                columns[f"share_mean{width}"] = (np.log(window.sum() / seoul.loc[window.index].sum())
                                                 if full else pd.Series(np.nan, index=history.columns))
            columns["share_lag0"] = share.loc[origin]
            for name, values in columns.items():
                frame[name] = values.reindex(units).to_numpy()

            def seoul_window(end, width):
                window = seoul.loc[end - (width - 1) * WEEK:end]
                return np.log(window.mean()) if len(window) == width and window.notna().all() else np.nan

            frame["seoul_lag0"] = log_seoul.get(origin, np.nan)
            frame["seoul_mean4"] = seoul_window(origin, 4)
            frame["seoul_mean13"] = seoul_window(origin, 13)
            frame["seoul_target_lag52"] = log_seoul.reindex(targets).to_numpy()
            pairs = pd.MultiIndex.from_arrays([targets, units])
            frame["share_target_lag52"] = share.stack(future_stack=True).reindex(pairs).to_numpy()
            level = frame.seoul_mean13 - seoul_window(origin - 52 * WEEK, 13)
            forecast = frame.seoul_target_lag52 + level
            frame[SHARE_OFFSET] = forecast.fillna(frame.seoul_mean13).fillna(frame.seoul_lag0)

            frame["anomaly_recent4"] = calendar.loc[origin - 3 * WEEK:origin].mean()
            frame["anomaly_recent13"] = calendar.loc[origin - 12 * WEEK:origin].mean()
            frame["anomaly_lag52"] = calendar.reindex(targets).fillna(0).to_numpy()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
