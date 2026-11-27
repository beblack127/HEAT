"""Align weekly mobility GT, daily NAVER trends, and raw subway exits."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "weekly_40"
OUT.mkdir(parents=True, exist_ok=True)
SUBWAY_RAW = ROOT / "work" / "Time-series-Project" / "data" / "subway_raw"
SUBWAY_MONTHLY = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly" / "subway_station_monthly_2023_2025.csv"


def normalize_station(name: str) -> str:
    return re.sub(r"\([^)]*\)|\s", "", name).strip()


def week_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("W-SUN").dt.start_time


def build() -> dict:
    prior = pd.read_csv(BASE / "age2030_pilot" / "aligned_panel_40.csv", dtype={"admin_dong_code": str})
    station_map = prior[["admin_dong_code", "station"]].drop_duplicates()
    if station_map.admin_dong_code.nunique() != 40:
        raise ValueError("Expected 40 unique station mappings")
    station_names = set(station_map.station)

    raw_files = sorted(SUBWAY_RAW.glob("CARD_SUBWAY_MONTH_20????.csv"))
    raw_files = [p for p in raw_files if "202301" <= p.stem[-6:] <= "202508"]
    if len(raw_files) != 32:
        raise ValueError(f"Expected 32 subway source files, found {len(raw_files)}")
    totals = defaultdict(int)
    for path in raw_files:
        for encoding in ("utf-8-sig", "cp949"):
            local = defaultdict(int)
            try:
                with path.open(encoding=encoding, newline="") as f:
                    for row in csv.DictReader(f):
                        station = normalize_station(row["역명"])
                        if station in station_names:
                            local[(row["사용일자"], station)] += int(row["하차총승객수"])
                for key, value in local.items():
                    totals[key] += value
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode subway source: {path.name}")
    subway = pd.DataFrame([{"date": d, "station": s, "subway_exits": v}
                           for (d, s), v in totals.items()])
    subway["date"] = pd.to_datetime(subway.date, format="%Y%m%d")
    subway.to_csv(OUT / "station_daily.csv", index=False, encoding="utf-8-sig")

    # Validate against the independently processed monthly station totals.
    monthly = pd.read_csv(SUBWAY_MONTHLY, encoding="utf-8-sig")
    monthly = monthly.loc[monthly.station_name.isin(station_names) & monthly.month.le("2025-08")]
    monthly = monthly.groupby(["station_name", "month"], as_index=False).exits.sum()
    check = subway.assign(month=subway.date.dt.strftime("%Y-%m")).groupby(["station", "month"], as_index=False).subway_exits.sum()
    check = check.merge(monthly, left_on=["station", "month"], right_on=["station_name", "month"], validate="one_to_one")
    if (check.subway_exits != check.exits).any():
        raise ValueError("Daily subway totals differ from monthly station source")

    mobility_files = sorted((BASE / "age2030_daily_baseline").glob("daily_457_age2030_20????.csv"))
    mobility = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                          for p in mobility_files], ignore_index=True)
    mobility = mobility.loc[mobility.admin_dong_code.isin(station_map.admin_dong_code)]
    mobility["date"] = pd.to_datetime(mobility.date.astype(str), format="%Y%m%d")
    panel = mobility[["admin_dong_code", "date", "age20_39"]].merge(station_map, on="admin_dong_code", validate="many_to_one")
    panel = panel.merge(subway, on=["date", "station"], validate="many_to_one")
    for age_label, folder in (("all", "daily_search_40"), ("age19_39", "daily_search_19to39_40")):
        search = pd.read_csv(BASE / folder / "dong_daily.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
        search["date"] = pd.to_datetime(search.date)
        panel = panel.merge(search.rename(columns={"search_index": f"search_{age_label}"}),
                            on=["admin_dong_code", "date"], validate="one_to_one")
    panel["week"] = week_start(panel.date)
    weekly = panel.groupby(["admin_dong_code", "week"], as_index=False).agg(
        y=("age20_39", "mean"), search_all=("search_all", "mean"),
        search_age19_39=("search_age19_39", "mean"), subway_exits=("subway_exits", "mean"),
        days=("date", "nunique"))
    weekly = weekly.loc[weekly.days.eq(7)].sort_values(["admin_dong_code", "week"])
    if len(weekly) != 40 * 139 or weekly[["y", "search_all", "search_age19_39", "subway_exits"]].isna().any().any():
        raise ValueError(f"Incomplete weekly panel: {len(weekly)} rows")
    weekly.to_csv(OUT / "weekly_panel_40.csv", index=False, encoding="utf-8-sig")
    audit = {"dongs": 40, "weeks": 139, "first_week": str(weekly.week.min().date()),
             "last_week": str(weekly.week.max().date()), "station_names": len(station_names),
             "daily_subway_matches_monthly": True,
             "gt": "weekly average daily 20-39 purpose 4+5+7 destination mobility",
             "subway": "nearest stable station daily exits, spatial proxy, not dong visitors"}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
