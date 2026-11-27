"""Check mobility-defined emerging candidates against independent search/transit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
RANK = BASE / "weekly_hotplace_rank"
OUT = RANK / "external_validity"
OUT.mkdir(parents=True, exist_ok=True)
HEAT = ROOT / "work" / "HEAT" / "data" / "processed"
STATION = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly" / "subway_station_monthly_2023_2025.csv"


def candidate_months(fold: str) -> pd.DataFrame:
    frame = pd.read_csv(RANK / f"{fold}_panel.csv", dtype={"admin_dong_code": str})
    frame["top20"] = False
    for _, group in frame.groupby("week"):
        eligible = group.loc[group.y_4w.ge(group.volume_floor)]
        top = eligible.nlargest(20, "actual_share_growth")
        frame.loc[top.index, "top20"] = True
    frame["month"] = frame.week.str[:7]
    monthly = frame.groupby(["admin_dong_code", "month"], as_index=False).agg(
        candidate_week_share=("top20", "mean"), actual_volume=("y_4w", "median"),
        floor=("volume_floor", "first"))
    monthly = monthly.loc[monthly.actual_volume.ge(monthly.floor)].copy()
    monthly["candidate"] = monthly.candidate_week_share.ge(.5)
    monthly["fold"] = fold
    return monthly


def main() -> None:
    monthly = pd.concat([candidate_months("validation"), candidate_months("test")], ignore_index=True)
    search = pd.read_csv(HEAT / "search" / "행정동_검색지수_월별.csv", encoding="utf-8-sig", dtype={"dong_code8": str})
    search = search.loc[search.tier.eq("전체"), ["dong_code8", "month", "seoul_share"]].sort_values(["dong_code8", "month"])
    search["search_share_yoy"] = search.seoul_share / search.groupby("dong_code8").seoul_share.shift(12) - 1
    monthly = monthly.merge(search[["dong_code8", "month", "search_share_yoy"]],
                            left_on=["admin_dong_code", "month"], right_on=["dong_code8", "month"],
                            how="left", validate="one_to_one")
    bridge = pd.read_csv(HEAT / "nametags" / "05_행정동_역_연결표.csv", encoding="utf-8-sig", dtype={"dong_code8": str})
    station = pd.read_csv(STATION, encoding="utf-8-sig")
    station = station.loc[station.month.le("2025-08")]
    station = station.groupby(["station_name", "month"], as_index=False).agg(exits=("exits", "sum"), days=("days", "max"))
    station["exits_daily"] = station.exits / station.days
    station["station_exits_yoy"] = station.exits_daily / station.groupby("station_name").exits_daily.shift(12) - 1
    complete = station.groupby("station_name").month.nunique()
    complete = set(complete.index[complete.eq(32)])
    bridge = bridge.loc[bridge.station.isin(complete)].sort_values(["dong_code8", "dist_to_dong_m"]).drop_duplicates("dong_code8")
    monthly = monthly.merge(bridge[["dong_code8", "station"]], on="dong_code8", how="left", validate="many_to_one")
    monthly = monthly.merge(station[["station_name", "month", "station_exits_yoy"]],
                            left_on=["station", "month"], right_on=["station_name", "month"],
                            how="left", validate="many_to_one")
    monthly.to_csv(OUT / "candidate_dong_month.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (fold, month, candidate), group in monthly.groupby(["fold", "month", "candidate"]):
        rows.append({"fold": fold, "month": month, "candidate": candidate, "n": len(group),
                     "search_coverage": round(float(group.search_share_yoy.notna().mean()), 3),
                     "search_yoy_median": round(float(group.search_share_yoy.median()), 3),
                     "search_positive_fraction": round(float(group.search_share_yoy.gt(0).mean()), 3),
                     "subway_yoy_median": round(float(group.station_exits_yoy.median()), 3)})
    summary = pd.DataFrame(rows).sort_values(["fold", "month", "candidate"])
    summary.to_csv(OUT / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    report = {"candidate_definition": "actual top20 four-week relative Seoul-share growth; above fixed 2023 volume floor; candidate in >=half evaluated weeks of month",
              "search_role": "retrospective external validity only; target-month search not used in forecasting",
              "subway_role": "retrospective nearby-station proxy; station may serve multiple dongs",
              "monthly_rows": len(monthly), "months": summary.month.nunique(),
              "test_search_median_candidate": float(monthly.loc[monthly.fold.eq("test") & monthly.candidate, "search_share_yoy"].median()),
              "test_search_median_control": float(monthly.loc[monthly.fold.eq("test") & ~monthly.candidate, "search_share_yoy"].median()),
              "test_subway_median_candidate": float(monthly.loc[monthly.fold.eq("test") & monthly.candidate, "station_exits_yoy"].median()),
              "test_subway_median_control": float(monthly.loc[monthly.fold.eq("test") & ~monthly.candidate, "station_exits_yoy"].median())}
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
