"""Evidence table for the two post-hoc 13-week forecast case slides."""
from collections import defaultdict
import csv
import io
from pathlib import Path
import tempfile
import zipfile

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "fixed_origin_search" / "case_diagnosis"
OUT.mkdir(exist_ok=True)
CODES = {"11500540": "Hwagok 1-dong", "11170640": "Ichon 2-dong"}

old = pd.read_csv(BASE / "fixed_origin_3month" / "backtest_predictions.csv",
                  dtype={"admin_dong_code": str})
new = pd.read_csv(BASE / "fixed_origin_search" / "backtest_search_predictions.csv",
                  dtype={"admin_dong_code": str})
pred = pd.concat([old, new], ignore_index=True)
pred = pred.loc[pred.admin_dong_code.isin(CODES)]
rows = []
for (code, model), g in pred.groupby(["admin_dong_code", "model"]):
    rows.append({"admin_dong_code": code, "area": CODES[code], "model": model,
                 "wape_pct": 100 * (g.actual - g.predicted).abs().sum() / g.actual.sum(),
                 "actual_mean": g.actual.mean(), "actual_min": g.actual.min(),
                 "actual_max": g.actual.max(), "forecast_mean": g.predicted.mean()})
pd.DataFrame(rows).sort_values(["admin_dong_code", "wape_pct"]).to_csv(
    OUT / "model_case_metrics.csv", index=False, encoding="utf-8-sig")

# Purpose-level monthly mean daily estimated trips for the selected dongs.
purpose_sums = defaultdict(float)
day_counts = defaultdict(set)
for month in ("202505", "202507"):
    archive_path = Path(tempfile.gettempdir()) / f"codex_seoul_mobility_{month}.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".csv"):
                continue
            with archive.open(member) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                header = next(reader)
                ix = {k: header.index(k) for k in
                      ("etl_ymd", "d_admdong_cd", "move_purpose", "male_20_cnt",
                       "feml_20_cnt", "male_30_cnt", "feml_30_cnt")}
                for row in reader:
                    code = row[ix["d_admdong_cd"]]
                    if code not in CODES:
                        continue
                    day = row[ix["etl_ymd"]]
                    purpose = row[ix["move_purpose"]]
                    day_counts[(month, code)].add(day)
                    age20_39 = sum(float(row[ix[c]]) for c in
                                   ("male_20_cnt", "feml_20_cnt", "male_30_cnt", "feml_30_cnt"))
                    purpose_sums[(month, code, purpose)] += age20_39

purpose_rows = []
for (month, code, purpose), value in purpose_sums.items():
    purpose_rows.append({"month": month, "admin_dong_code": code,
                         "area": CODES[code], "purpose": purpose,
                         "daily_mean_age20_39": value / len(day_counts[(month, code)])})
pd.DataFrame(purpose_rows).sort_values(["admin_dong_code", "purpose", "month"]).to_csv(
    OUT / "purpose_may_july.csv", index=False, encoding="utf-8-sig")

# Exact 20s+30s mobility on the two annual fireworks festival dates.
event_rows = []
for month, day in (("202310", "20231007"), ("202410", "20241005")):
    archive_path = Path(tempfile.gettempdir()) / f"codex_seoul_mobility_{month}.zip"
    member = f"seoul_purpose_admdong1_in_{day}.csv"
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        header = next(reader)
        ix = {k: header.index(k) for k in
              ("d_admdong_cd", "move_purpose", "male_20_cnt", "feml_20_cnt",
               "male_30_cnt", "feml_30_cnt")}
        for row in reader:
            if row[ix["d_admdong_cd"]] != "11170640" or row[ix["move_purpose"]] not in {"4", "5", "7"}:
                continue
            event_rows.append({"date": day, "purpose": row[ix["move_purpose"]],
                               "estimated_trips_age20_39": sum(float(row[ix[c]]) for c in
                               ("male_20_cnt", "feml_20_cnt", "male_30_cnt", "feml_30_cnt"))})
pd.DataFrame(event_rows).to_csv(OUT / "festival_dates_purpose.csv", index=False, encoding="utf-8-sig")

print(pd.read_csv(OUT / "purpose_may_july.csv").to_string(index=False))
