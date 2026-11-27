"""Aggregate Seoul OA-22298 daily archives to monthly 20s+30s 4+5+7 arrivals."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_mobility"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly"


def aggregate(month: str) -> dict:
    archive = Path(tempfile.gettempdir()) / f"codex_seoul_mobility_{month}.zip"
    if not zipfile.is_zipfile(archive):
        raise FileNotFoundError(f"OA-22298 archive missing or invalid: {archive}")
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    dates = set()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with z.open(member) as raw:
                reader = csv.reader(__import__("io").TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                header = next(reader)
                ix = {name: header.index(name) for name in (
                    "d_admdong_cd", "move_purpose", "male_20_cnt", "male_30_cnt",
                    "feml_20_cnt", "feml_30_cnt", "total_cnt", "etl_ymd")}
                for row in reader:
                    if row[ix["move_purpose"]] not in {"4", "5", "7"}:
                        continue
                    code = row[ix["d_admdong_cd"]]
                    if len(code) != 8 or not code.startswith("11"):
                        continue
                    dates.add(row[ix["etl_ymd"]])
                    a = sums[code]
                    a[0] += float(row[ix["male_20_cnt"]]) + float(row[ix["feml_20_cnt"]])
                    a[1] += float(row[ix["male_30_cnt"]]) + float(row[ix["feml_30_cnt"]])
                    a[2] += float(row[ix["total_cnt"]])
                    a[3] += sum(float(row[ix[c]]) for c in ("male_20_cnt", "male_30_cnt", "feml_20_cnt", "feml_30_cnt"))
                    a[4] += 1
    if len(dates) < 27:
        raise ValueError(f"Incomplete month {month}: {len(dates)} dates")
    if not sums:
        raise ValueError(f"No destination dongs in {month}")
    rows = []
    for code, a in sorted(sums.items()):
        rows.append({
            "month": f"{month[:4]}-{month[4:]}", "admin_dong_code": code, "days": len(dates),
            "age20_daily": a[0] / len(dates), "age30_daily": a[1] / len(dates),
            "mobility_457_age20_39_daily": a[3] / len(dates),
            "mobility_457_allages_daily": a[2] / len(dates),
            "age20_39_share": a[3] / a[2] if a[2] else "", "records": a[4],
        })
    path = OUT / f"age2030_457_{month}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"month": month, "days": len(dates), "dongs": len(rows), "total_age20_39_daily": round(sum(r["mobility_457_age20_39_daily"] for r in rows), 2),
             "total_457_daily": round(sum(r["mobility_457_allages_daily"] for r in rows), 2)}
    original = SOURCES / f"mobility_{month}.csv"
    if original.exists():
        with original.open(encoding="utf-8-sig", newline="") as f:
            old = {r["admin_dong_code"]: float(r["mobility_shop_tour_other_daily"]) for r in csv.DictReader(f)}
        new = {r["admin_dong_code"]: r["mobility_457_allages_daily"] for r in rows}
        if not set(old).issubset(new):
            raise ValueError(f"Missing legacy pilot dongs in {month}")
        delta = max(abs(old[c] - new[c]) for c in old)
        audit["legacy_40_max_abs_difference"] = delta
        if delta > 1e-5:
            raise ValueError(f"Age source disagrees with existing mobility aggregate: {month}, {delta}")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="202301")
    parser.add_argument("--end", default="202508")
    args = parser.parse_args()
    months = [f"{y}{m:02d}" for y in range(2023, 2026) for m in range(1, 13)]
    audits = []
    for month in months:
        if args.start <= month <= args.end:
            audit = aggregate(month)
            audits.append(audit)
            print(json.dumps(audit, ensure_ascii=False), flush=True)
    (OUT / "aggregation_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
