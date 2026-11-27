"""Korean public holidays, semester starts, and anomaly windows used by the v2 experiments.

Holiday lists follow the official public-holiday calendar (including substitute and temporary
holidays). The 2026 entries cover only the forecast period; check them against the official
calendar before the final run.
"""

import pandas as pd

SEOLLAL = pd.to_datetime([
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2025-01-28", "2025-01-29", "2025-01-30",
    "2026-02-16", "2026-02-17", "2026-02-18",
])
CHUSEOK = pd.to_datetime([
    "2023-09-28", "2023-09-29", "2023-09-30",
    "2024-09-16", "2024-09-17", "2024-09-18",
    "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
])
OTHER_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-10-02", "2023-10-03", "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15",
    "2024-06-06", "2024-08-15", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06",
    "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-03-01", "2026-03-02",
])
HOLIDAYS = SEOLLAL.union(CHUSEOK).union(OTHER_HOLIDAYS)
FESTIVALS = SEOLLAL.union(CHUSEOK)
# The festival day itself (Seollal / Chuseok day); its Monday–Sunday week is the anchor week.
FESTIVAL_DAYS = {
    "seollal": pd.to_datetime(["2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17"]),
    "chuseok": pd.to_datetime(["2022-09-10", "2023-09-29", "2024-09-17", "2025-10-06"]),
}
# First weekday of March that is not a holiday.
SEMESTER_START = pd.to_datetime(["2023-03-02", "2024-03-04", "2025-03-04", "2026-03-03"])
# Fall semester: first weekday of September that is not a holiday.
FALL_START = pd.to_datetime(["2023-09-01", "2024-09-02", "2025-09-01"])
TERM_WEEKS = 16

# Data anomalies identified from the daily index (2026_modeling_preprocessing_features.md, D-1).
WINDOWS = {
    "summer2024": (pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-31")),
    "jan2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
    "spring2025": (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30")),
}
# Windows whose start and end coincide with month boundaries; excluded from the clean evaluation.
CONFIRMED = ("summer2024", "spring2025")
# Normal periods used to scale the expected Seoul level from the same months of the previous year.
GROWTH = {
    "summer2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-31")),
    "jan2025": (pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31")),
    "spring2025": (pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31")),
}


def week_end(dates):
    dates = pd.DatetimeIndex(dates)
    return dates + pd.to_timedelta((6 - dates.weekday) % 7, unit="D")


def festival_weeks():
    """Sundays of Monday–Sunday weeks containing a Seollal or Chuseok holiday."""
    return pd.DatetimeIndex(week_end(FESTIVALS).unique())


def window_days(names):
    days = [pd.date_range(*WINDOWS[name]) for name in names]
    return pd.DatetimeIndex([]).append(days).unique() if days else pd.DatetimeIndex([])


def week_calendar(weeks):
    """Per Sunday: holiday counts, festival flags, semester start, and anomaly-day counts."""
    weeks = pd.DatetimeIndex(weeks)
    frame = pd.DataFrame(index=weeks)
    for name, days in (("holiday_days", HOLIDAYS), ("seollal_days", SEOLLAL), ("chuseok_days", CHUSEOK)):
        frame[name] = pd.Series(1, index=week_end(days)).groupby(level=0).sum().reindex(weeks, fill_value=0)
    frame["semester_start"] = weeks.isin(week_end(SEMESTER_START))
    for name in WINDOWS:
        frame[f"{name}_days"] = (pd.Series(1, index=week_end(window_days([name])))
                                 .groupby(level=0).sum().reindex(weeks, fill_value=0))
    frame["confirmed_anomaly_days"] = frame[[f"{name}_days" for name in CONFIRMED]].sum(axis=1)
    return frame
