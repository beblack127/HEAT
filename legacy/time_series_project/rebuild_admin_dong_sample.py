"""Build a provisional, spatially consistent Seoul administrative-dong panel.

The 100 dongs are a reproducible sampling frame, not hotspot/control labels.
Existing Naver series are reused only for unique, exact administrative-dong
name matches. Search intent still requires a separate semantic review.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data/reference/seoul_admin_dong_codes_official.zip"
OLD_AREAS = ROOT / "data/naver/area_list_100_planned.csv"
OLD_SEARCH = ROOT / "data/naver/areas_100_20261106T105913Z/monthly_search_interest_100areas.csv"
OUT = ROOT / "data/admin_dong_sample"
DB = OUT / "admin_dong_100_provisional.sqlite"
MONTHS = [f"{y}-{m:02d}" for y in range(2023, 2026) for m in range(1, 13)]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        by_month = {}
        for month in MONTHS:
            ym = month.replace("-", "")
            raw = archive.read(f"SEOUL_ADMI/ADMI_{ym}.csv")
            rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            by_month[month] = {
                row["ADMI_CD"]: row
                for row in rows
                if row["SIDO_NM"] == "서울특별시"
            }

    current = by_month[MONTHS[-1]]
    stable = {
        code: row for code, row in current.items()
        if all(code in by_month[month]
               and by_month[month][code]["ADMI_NM"] == row["ADMI_NM"]
               and by_month[month][code]["SGG_NM"] == row["SGG_NM"]
               for month in MONTHS)
    }
    name_count = Counter(row["ADMI_NM"] for row in current.values())
    # A bare query cannot identify one dong if multiple Seoul dongs share its name.
    eligible = {code: row for code, row in stable.items()
                if name_count[row["ADMI_NM"]] == 1}
    by_name = {row["ADMI_NM"]: row for row in eligible.values()}

    old_areas = read_csv(OLD_AREAS)
    candidates = defaultdict(list)
    for old in old_areas:
        if old["search_term"] in by_name:
            candidates[old["search_term"]].append(old)
    seed = {by_name[name]["ADMI_CD"]: old[0]
            for name, old in candidates.items() if len(old) == 1}

    districts = sorted({r["SGG_NM"] for r in eligible.values()})
    assert len(districts) == 25
    selected = {}
    for district in districts:
        pool = [r for r in eligible.values() if r["SGG_NM"] == district]
        seed_rows = [r for r in pool if r["ADMI_CD"] in seed]
        if len(seed_rows) > 4:
            raise ValueError(f"More than four existing exact matches in {district}")
        remainder = [r for r in pool if r["ADMI_CD"] not in seed]
        remainder.sort(key=lambda r: hashlib.sha256(
            f"heat-admin-dong-v1:{r['ADMI_CD']}".encode()).hexdigest())
        for row in seed_rows + remainder[:4 - len(seed_rows)]:
            selected[row["ADMI_CD"]] = row
    assert len(selected) == 100
    assert set(seed).issubset(selected)

    sample = []
    for code, row in sorted(selected.items(), key=lambda item: (item[1]["SGG_NM"], item[1]["ADMI_NM"])):
        previous = seed.get(code)
        sample.append({
            "admin_dong_code": code,
            "district": row["SGG_NM"],
            "admin_dong_name": row["ADMI_NM"],
            "official_full_name": row["FULL_NM"],
            "sample_rule": "existing_exact_name" if previous else "district_balanced_hash",
            "search_area_original": previous["search_area"] if previous else "",
            "search_term": previous["search_term"] if previous else row["ADMI_NM"],
            "naver_status": "existing_exact_name_proxy_needs_review" if previous else "not_collected",
            "hotspot_group": "unassigned_requires_baseline_activity",
        })
    write_csv(OUT / "sample_100_admin_dongs.csv", list(sample[0]), sample)

    search = {(r["search_area"], r["month"]): r for r in read_csv(OLD_SEARCH)}
    panel = []
    for row in sample:
        for month in MONTHS:
            found = search.get((row["search_area_original"], month)) if row["search_area_original"] else None
            panel.append({
                "admin_dong_code": row["admin_dong_code"],
                "month": month,
                "district": row["district"],
                "admin_dong_name": row["admin_dong_name"],
                "search_interest": found["search_interest"] if found else "",
                "search_data_status": "existing_name_proxy" if found and found["search_interest"] else "not_collected",
                "search_area_original": row["search_area_original"],
                "visitor_activity": "",
                "footfall": "",
                "living_population": "",
                "food_openings": "",
                "food_closures": "",
                "subway_alightings": "",
            })
    write_csv(OUT / "monthly_panel_100_admin_dongs_provisional.csv", list(panel[0]), panel)

    with sqlite3.connect(DB) as conn:
        conn.execute("DROP TABLE IF EXISTS admin_dong_sample")
        conn.execute("DROP TABLE IF EXISTS monthly_panel")
        conn.execute("""CREATE TABLE admin_dong_sample (
          admin_dong_code TEXT PRIMARY KEY, district TEXT, admin_dong_name TEXT,
          official_full_name TEXT, sample_rule TEXT, search_area_original TEXT,
          search_term TEXT, naver_status TEXT, hotspot_group TEXT)""")
        conn.execute("""CREATE TABLE monthly_panel (
          admin_dong_code TEXT, month TEXT, district TEXT, admin_dong_name TEXT,
          search_interest REAL, search_data_status TEXT, search_area_original TEXT,
          visitor_activity REAL, footfall REAL, living_population REAL,
          food_openings INTEGER, food_closures INTEGER, subway_alightings INTEGER,
          PRIMARY KEY(admin_dong_code, month),
          FOREIGN KEY(admin_dong_code) REFERENCES admin_dong_sample(admin_dong_code))""")
        conn.executemany("INSERT INTO admin_dong_sample VALUES (?,?,?,?,?,?,?,?,?)",
                         [tuple(r.values()) for r in sample])
        conn.executemany("INSERT INTO monthly_panel VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         [tuple(None if value == "" else value for value in r.values())
                          for r in panel])
        conn.execute("""CREATE VIEW IF NOT EXISTS observed_search_panel AS
          SELECT * FROM monthly_panel WHERE search_interest IS NOT NULL""")
        count = conn.execute("SELECT COUNT(*), COUNT(search_interest) FROM monthly_panel").fetchone()
    print(f"Official stable unique-name dongs: {len(eligible)}")
    print(f"Sample: {len(sample)} dongs, four per district")
    print(f"Existing Naver name matches: {len(seed)} dongs")
    print(f"Panel: {count[0]} rows, {count[1]} existing search observations")
    print(DB)


if __name__ == "__main__":
    main()
