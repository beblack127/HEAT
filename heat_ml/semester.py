"""University calendar features for v2 step 3 (preprocessing doc B-3; catalog F6, F7).

F6  Target-week calendar: in_semester (at least 3 weekdays inside a 16-week term starting at the
    spring or fall semester start), semester_start (week containing the March start),
    term_week (1..16 within a term, 0 outside).
F7  Dong semester amplitude as of the origin: log(mean Seoul share in term months / mean share in
    vacation months), term = Mar–May and Sep–Nov, vacation = Jan–Feb and Jul–Aug, festival weeks
    excluded, share = dong / 426-dong sum of the corrected history. The doc's March / Jan–Feb
    ratio is widened to both terms so each origin has more than one observed term.
"""

import numpy as np
import pandas as pd

from .data import WEEK
from .kr_calendar import FALL_START, SEMESTER_START, TERM_WEEKS, festival_weeks, week_end

SEMESTER_GROUPS = {
    "semester": ("in_semester", "semester_start", "term_week"),
    "semester_amp": ("semester_amp", "semester_amp_x_term", "semester_amp_x_start"),
}
TERM_MONTHS = (3, 4, 5, 9, 10, 11)
VACATION_MONTHS = (1, 2, 7, 8)


def semester_calendar(weeks):
    weeks = pd.DatetimeIndex(weeks)
    days = pd.date_range(weeks[0] - pd.Timedelta(days=6), weeks[-1])
    weekdays = days[days.weekday < 5]
    term_week = pd.Series(0, index=weekdays)
    for start in SEMESTER_START.union(FALL_START):
        inside = (weekdays >= start) & (weekdays < start + TERM_WEEKS * WEEK)
        term_week[inside] = ((weekdays[inside] - start).days // 7 + 1)
    by_week = term_week.groupby(week_end(weekdays))
    frame = pd.DataFrame(index=weeks)
    frame["in_semester"] = by_week.apply(lambda s: float(s.gt(0).sum() >= 3)).reindex(weeks, fill_value=0.0)
    frame["term_week"] = by_week.max().reindex(weeks, fill_value=0).astype(float) * frame.in_semester
    frame["semester_start"] = weeks.isin(week_end(SEMESTER_START)).astype(float)
    return frame


def semester_amplitude(history):
    share = history.div(history.sum(axis=1), axis=0)
    share = share.loc[~share.index.isin(festival_weeks())]
    months = share.index.month
    in_term, in_vacation = months.isin(TERM_MONTHS), months.isin(VACATION_MONTHS)
    if in_term.sum() < 4 or in_vacation.sum() < 4:
        return pd.Series(np.nan, index=history.columns)
    return np.log(share.loc[in_term].mean() / share.loc[in_vacation].mean())


class SemesterMixin:
    """Adds the university calendar columns to Panel.examples for any Panel subclass."""

    def examples(self, origins, label_cutoff=None):
        calendar = getattr(self, "_semester_calendar", None)
        if calendar is None:
            calendar = self._semester_calendar = semester_calendar(self.weeks)
        frames = []
        for origin in origins:
            frame = super().examples([origin], label_cutoff)
            if frame.empty:
                frames.append(frame)
                continue
            info = calendar.reindex(frame.week_end)
            for column in SEMESTER_GROUPS["semester"]:
                frame[column] = info[column].to_numpy()
            amplitude = semester_amplitude(self.corrected(origin))
            frame["semester_amp"] = amplitude.reindex(frame.unit).to_numpy()
            frame["semester_amp_x_term"] = frame.semester_amp * frame.in_semester
            frame["semester_amp_x_start"] = frame.semester_amp * frame.semester_start
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
