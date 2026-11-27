"""Read and harmonize mobility and optionally released covariates."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "HEAT/data/processed"
NAMES = PROCESSED / "nametags"
DONGS = NAMES / "01_행정동_목록.csv"
SEARCH = PROCESSED / "search/행정동_검색지수_월별.csv"
FNB = PROCESSED / "fnb/행정동_개폐업_월별.csv"
SUBWAY = ROOT / "HEAT/data/raw/subway/subway_daily.parquet"
COMMON = ROOT / "HEAT/results/gt/unit_metrics.parquet"
MERGE = {"11230515": "11230536", "11230533": "11230536"}
MOBILITY_MAP = {**MERGE, "11680740": "11680675"}
START = pd.Timestamp("2023-01-08")
FINAL = pd.Timestamp("2025-12-28")
HORIZON = 13
WEEK = pd.Timedelta(weeks=1)
ORIGINS = tuple(pd.to_datetime(["2024-10-27", "2025-01-26", "2025-04-27", "2025-07-27"]))
CONDITIONS = {
    "base": (), "search": ("search",), "fnb": ("fnb",), "subway": ("subway",),
    "all": ("search", "fnb", "subway"),
}


def read(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={k: str for k in ("unit", "dong_code8", "code")})


def sundays(dates):
    return dates + pd.to_timedelta((6 - dates.dt.weekday) % 7, unit="D")


def load_target(path):
    frame = read(path)
    frame["unit"] = frame.unit.astype(str)
    frame["week_end"] = pd.to_datetime(frame.week_end)
    if frame.duplicated(["unit", "week_end"]).any() or not frame.week_end.dt.weekday.eq(6).all():
        raise ValueError("Target requires unique unit × Sunday keys.")
    frame.loc[frame.n_days.lt(4), "y"] = np.nan
    if (frame[["y", "seoul_y"]] < 0).any().any():
        raise ValueError("Mobility must be nonnegative.")
    if frame.groupby("week_end").seoul_y.nunique().gt(1).any():
        raise ValueError("seoul_y must be constant within each week.")
    return frame.loc[frame.week_end.ge(START)]


def read_daily_mobility(directory, days):
    """Aggregate the selected dates by original Seoul destination code."""
    ages = ["male_20_cnt", "male_30_cnt", "feml_20_cnt", "feml_30_cnt"]
    keys = ["d_admdong_cd", "move_purpose", "etl_ymd"]
    daily = []
    for month in days.to_period("M").unique():
        path = directory / f"seoul_purpose_admdong1_in_{month.strftime('%Y%m')}.zip"
        print(f"Reading {path.name}", flush=True)
        with ZipFile(path) as archive:
            names = {Path(name).stem[-8:]: name for name in archive.namelist() if name.endswith(".csv")}
            for date in days[days.to_period("M") == month]:
                key = date.strftime("%Y%m%d")
                with archive.open(names[key]) as stream:
                    raw = pd.read_csv(stream, usecols=keys + ages,
                                      dtype={"d_admdong_cd": str, "etl_ymd": str})
                if not raw.etl_ymd.eq(key).all():
                    raise ValueError(f"Unexpected observation date in {names[key]}")
                raw = raw.loc[raw.d_admdong_cd.str.startswith("11", na=False)]
                if raw.empty:
                    raise ValueError(f"No Seoul destination rows in {names[key]}")
                movement = raw[ages].sum(axis=1).where(raw.move_purpose.isin([4, 5, 7]), 0)
                daily.append(raw.assign(y=movement).groupby("d_admdong_cd", as_index=False)
                             .y.sum().rename(columns={"d_admdong_cd": "unit"}).assign(date=date))
    return pd.concat(daily, ignore_index=True)


def prepare_mobility(directory, first, last):
    """Reproduce HEAT's daily aggregation, including zero selected purposes."""
    days = pd.date_range(first, last)
    if days.empty or days[0].weekday() != 0 or days[-1].weekday() != 6:
        raise ValueError("Prepare requires a Monday start and a Sunday end.")
    frame = read_daily_mobility(directory, days)
    frame["week_end"] = sundays(frame.date)
    seoul = frame.groupby(["week_end", "date"]).y.sum().groupby(level=0).agg(["mean", "size"])
    seoul = seoul.loc[seoul["size"].eq(7), "mean"].rename("seoul_y")
    frame["unit"] = frame.unit.replace(MOBILITY_MAP)
    allowed = read(DONGS).dong_code8.replace(MERGE)
    daily_units = frame.loc[frame.unit.isin(allowed)].groupby(["unit", "week_end", "date"]).y.sum()
    weekly = daily_units.groupby(level=[0, 1]).agg(y="mean", n_days="size").reset_index()
    weekly.loc[weekly.n_days.lt(4), "y"] = np.nan
    return weekly.merge(seoul, on="week_end")


def common_units(path, units):
    if path is None:
        return None
    frame = read(path)
    frame["unit"] = frame.unit.astype(str)
    if "origin" in frame:
        frame["origin"] = pd.to_datetime(frame.origin)
        membership = {o: set(frame.loc[frame.origin.eq(o), "unit"]) for o in ORIGINS}
    else:
        membership = dict.fromkeys(ORIGINS, set(frame.unit))
    if any(len(ids) != 416 or not ids <= set(units) for ids in membership.values()):
        raise ValueError("Expected 416 distinct known HEAT units at each outer origin.")
    return membership


def monthly(frame, columns, calendar):
    frame["period"] = pd.PeriodIndex(frame.month, freq="M").to_timestamp("M")
    grouped = frame.groupby(["unit", "period"])[columns].sum(min_count=1)
    return grouped.reindex(pd.MultiIndex.from_product([frame.unit.unique(), calendar],
                                                      names=["unit", "period"]))


def load_covariates(paths, releases, *, assume_period_end=False):
    """Map source values to available dates, including every rolling dependency."""
    result = {}
    if releases is None and not assume_period_end:
        return result
    dates = None
    if releases is not None:
        dates = read(releases)
        dates["period"] = pd.to_datetime(dates.period)
        dates["available_on"] = pd.to_datetime(dates.available_on)
        if dates.duplicated(["feature", "period"]).any() or dates.available_on.lt(dates.period).any():
            raise ValueError("Release dates require unique feature/period keys and cannot precede period end.")

    for name, path in paths.items():
        if path is None:
            continue
        release = None if dates is None else dates.loc[dates.feature.eq(name)].set_index("period").available_on
        if release is not None and release.dropna().empty:
            continue
        source = read(path)
        if name == "search":
            source = source.loc[source.tier.eq("전체")].copy()
            source["unit"] = source.dong_code8.replace(MERGE)
            source["period"] = pd.PeriodIndex(source.month, freq="M").to_timestamp("M")
            share = source.groupby(["unit", "period"]).seoul_share.sum(min_count=1)
            value = np.log(share.where(share.gt(0)) * 1e4)
            delays = (0,)
        elif name == "fnb":
            source = source.loc[source.level.eq("dong")].copy()
            source["unit"] = source.code.replace(MERGE)
            months = pd.PeriodIndex(source.month, freq="M")
            calendar = pd.period_range(months.min(), months.max(), freq="M").to_timestamp("M")
            source = monthly(source, ["개업", "폐업_일괄제외", "영업점포"], calendar)
            net = source["개업"] - source["폐업_일괄제외"]
            stock = source["영업점포"].groupby(level=0).shift(3)
            value = net.groupby(level=0).transform(lambda x: x.rolling(3).sum()) / stock.where(stock.gt(0))
            delays = (0, 1, 2, 3)
        else:
            source["date"] = pd.to_datetime(source.date)
            source["period"] = sundays(source.date)
            links = read(NAMES / "05_행정동_역_연결표.csv")
            links["station"] = links.station.replace({"서울역": "서울"})
            links = links.groupby(["station", "dong_code8"], as_index=False).share_1000.max()
            land = read(DONGS).set_index("dong_code8").land_km2
            links["weight"] = links.share_1000 * links.dong_code8.map(land)
            links = links.loc[links.weight.gt(0)].copy()
            links["weight"] /= links.groupby("station").weight.transform("sum")
            links["unit"] = links.dong_code8.replace(MERGE)
            station = source.groupby(["station", "period"], as_index=False).alight.mean()
            dong = station.merge(links, on="station")
            counts = dong.assign(weighted=dong.alight * dong.weight).groupby(["unit", "period"]).weighted.sum()
            calendar = pd.date_range(source.period.min(), source.period.max(), freq="W-SUN")
            counts = counts.reindex(pd.MultiIndex.from_product([links.unit.unique(), calendar],
                                                               names=["unit", "period"]))
            value = np.log1p(counts.groupby(level=0).transform(lambda x: x.rolling(4).mean()))
            delays = (0, 1, 2, 3)
        period = value.index.get_level_values("period")
        if release is None:
            available = period
        else:
            dependencies = [release.reindex(period - (pd.offsets.MonthEnd(i) if name != "subway" else i * WEEK))
                            .to_numpy() for i in delays]
            release_frame = pd.DataFrame(dict(enumerate(dependencies)), index=value.index)
            available = release_frame.max(axis=1, skipna=False)
        result[name] = pd.DataFrame({name: value, f"{name}_available": available})
    return result

