"""Dong and week segments for v2 evaluation (preprocessing doc, section C).

Dong types are an evaluation grouping only, never a model input, so they are computed once from
raw weekly Seoul shares for 2023-01-08..2025-12-28 with every candidate anomaly week masked.
Model features of this kind (doc B-1b, B-3, B-6) must instead be recomputed at each origin.
Priority when a dong meets several rules: university > step > festival_residential > stable.
University comes first because the vacation/semester swing itself looks like a level step.
"""

import numpy as np
import pandas as pd

from .kr_calendar import WINDOWS, festival_weeks, week_calendar

STEP_MIN = 0.30          # |median of next 13 weeks / median of previous 13 weeks − 1|
SEMESTER_MIN = 1.20      # March share / Jan–Feb share (2023, 2024 mean)
FESTIVAL_MIN = 1.20      # festival-week share / share 2–3 weeks either side (median over festivals)


def dong_types(target):
    weekly = target.pivot(index="week_end", columns="unit", values="y")
    seoul = target.groupby("week_end").seoul_y.first()
    calendar = week_calendar(weekly.index)
    anomalous = calendar[[f"{name}_days" for name in WINDOWS]].sum(axis=1).gt(0)
    festivals = festival_weeks()
    share = weekly.div(seoul, axis=0).where(~anomalous, axis=0)  # mask, do not bridge the gap

    log_share = np.log(share.loc[~share.index.isin(festivals)])
    before = log_share.rolling(13, min_periods=10).median()
    after = log_share[::-1].rolling(13, min_periods=10).median()[::-1].shift(-1)
    step = np.expm1((after - before).abs().max())

    amplitude = []
    for year in (2023, 2024):
        winter = share.loc[f"{year}-01-01":f"{year}-02-28"]
        winter = winter.loc[~winter.index.isin(festivals)]
        amplitude.append(share.loc[f"{year}-03-06":f"{year}-03-31"].mean() / winter.mean())
    semester = pd.concat(amplitude, axis=1).mean(axis=1)

    ratios = []
    for week in festivals:
        if week not in share.index:
            continue
        around = [week + k * pd.Timedelta(weeks=1) for k in (-3, -2, 2, 3)]
        around = [day for day in around if day in share.index and day not in festivals]
        if share.loc[[week, *around]].isna().all(axis=1).any():
            continue
        ratios.append(share.loc[week] / share.loc[around].mean())
    festival = pd.concat(ratios, axis=1).median(axis=1)

    frame = pd.DataFrame({"step_change": step, "semester_amplitude": semester, "festival_ratio": festival})
    frame["dong_type"] = np.select(
        [frame.semester_amplitude.ge(SEMESTER_MIN), frame.step_change.ge(STEP_MIN),
         frame.festival_ratio.ge(FESTIVAL_MIN)],
        ["university", "step", "festival_residential"], "stable")
    return frame.rename_axis("unit").reset_index()


def week_types(weeks):
    calendar = week_calendar(weeks)
    kind = np.select(
        [calendar.seollal_days.gt(0), calendar.chuseok_days.gt(0), calendar.semester_start,
         calendar.holiday_days.gt(0)],
        ["seollal", "chuseok", "semester_start", "other_holiday"], "normal")
    return calendar.assign(week_type=kind).rename_axis("week_end").reset_index()
