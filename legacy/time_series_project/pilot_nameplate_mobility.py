"""Audit the supplied nameplate ZIP against existing monthly mobility summaries.

Run: python pilot_nameplate_mobility.py --nameplate PATH_TO_ZIP --out OUTPUT_DIR
This is a baseline pilot. It does not collect or impute Naver search values.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def zip_csv(archive: zipfile.ZipFile, prefix: str):
    matches = [n for n in archive.namelist()
               if n.split("/")[-1].startswith(prefix) and n.lower().endswith(".csv")
               and not n.startswith("__MACOSX/") and "/._" not in n]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix} CSV in ZIP; found {len(matches)}")
    return list(csv.DictReader(io.StringIO(archive.read(matches[0]).decode("utf-8-sig"))))


def month_offset(month: str, delta: int) -> str:
    year, m = map(int, month.split("-"))
    n = year * 12 + m - 1 + delta
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def metrics(rows, field):
    errors = [abs(float(r["actual_daily"]) - float(r[field])) for r in rows]
    total = sum(float(r["actual_daily"]) for r in rows)
    return {"n": len(rows), "mae": round(statistics.mean(errors), 2),
            "wape": round(sum(errors) / total, 4) if total else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nameplate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.nameplate) as archive:
        dongs = {r["dong_code8"]: r for r in zip_csv(archive, "01_")}
        links = zip_csv(archive, "04_")
    if len(dongs) != 427:
        raise ValueError(f"Unexpected nameplate dong count: {len(dongs)}")

    source_files = sorted((ROOT / "data/context_monthly").glob("mobility_20????.csv"))
    source_rows = [r for p in source_files for r in read_csv(p)]
    if not source_rows:
        raise ValueError("No monthly mobility summary files found")
    keyed = {}
    for r in source_rows:
        key = (r["admin_dong_code"], r["month"])
        if key in keyed:
            raise ValueError(f"Duplicate dong-month: {key}")
        keyed[key] = r
    codes = {code for code, _ in keyed}
    unmatched = sorted(codes - dongs.keys())
    if unmatched:
        raise ValueError(f"Mobility codes absent from nameplate: {unmatched}")

    by_slot = defaultdict(list)
    for r in links:
        if r["usable"] == "1":
            by_slot[(r["dong_code8"], r["tier"])].append(r)
    for key, rows in by_slot.items():
        if abs(sum(float(r["weight"]) for r in rows) - 1) > 0.001:
            raise ValueError(f"Nameplate weights do not sum to one: {key}")

    coverage = []
    for code in sorted(codes):
        dr = dongs[code]
        coverage.append({"dong_code8": code, "gu": dr["gu"],
                         "dong_name": dr["dong_name"],
                         **{f"usable_{tier}": int((code, tier) in by_slot)
                            for tier in ("동네", "역", "권역")},
                         **{f"n_{tier}_names": len(by_slot[(code, tier)])
                            for tier in ("동네", "역", "권역")}})

    predictions = []
    for (code, month), r in sorted(keyed.items()):
        if not month.startswith("2025-"):
            continue
        lag2 = keyed.get((code, month_offset(month, -2)))
        lag12 = keyed.get((code, month_offset(month, -12)))
        if not lag2 or not lag12:
            continue
        actual = float(r["mobility_shop_tour_other_daily"])
        shop_tour = float(r["mobility_shop_tour_daily"])
        predictions.append({"dong_code8": code, "month": month,
                            "actual_daily": round(actual, 4),
                            "shop_tour_daily": round(shop_tour, 4),
                            "other_share": round((actual-shop_tour)/actual, 4) if actual else "",
                            "lag2_daily": round(float(lag2["mobility_shop_tour_other_daily"]), 4),
                            "lag12_daily": round(float(lag12["mobility_shop_tour_other_daily"]), 4),
                            "target_days": r["days"]})

    def write_csv(name, rows):
        with (args.out / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv("nameplate_mobility_coverage.csv", coverage)
    write_csv("mobility_baseline_2025.csv", predictions)
    pre_notice = [r for r in predictions if r["month"] < "2025-09"]
    post_notice = [r for r in predictions if r["month"] >= "2025-09"]
    target_total = sum(float(r["actual_daily"]) for r in predictions)
    shop_tour_total = sum(float(r["shop_tour_daily"]) for r in predictions)
    result = {"nameplate_zip": args.nameplate.name,
              "source_months": sorted({m for _, m in keyed}),
              "mobility_dongs": len(codes), "nameplate_dongs": len(dongs),
              "matched_dongs": len(codes), "source_rows": len(keyed),
              "shop_tour_nonzero_rows": sum(float(r["mobility_shop_tour_daily"]) > 0 for r in source_rows),
              "shop_tour_nonzero_rows_2025": sum(float(r["shop_tour_daily"]) > 0 for r in predictions),
              "shop_tour_share_2025": round(shop_tour_total / target_total, 4),
              "coverage": {tier: sum(r[f"usable_{tier}"] for r in coverage)
                           for tier in ("동네", "역", "권역")},
              "evaluation_2025": {field: metrics(predictions, field)
                                  for field in ("lag2_daily", "lag12_daily")},
              "evaluation_2025_jan_aug": {field: metrics(pre_notice, field)
                                          for field in ("lag2_daily", "lag12_daily")},
              "evaluation_2025_sep_dec": {field: metrics(post_notice, field)
                                          for field in ("lag2_daily", "lag12_daily")},
              "other_share_median_2025": round(statistics.median(float(r["other_share"])
                                                        for r in predictions if r["other_share"] != ""), 4),
              "notes": ["Existing mobility summaries cover 40 selected dongs, not all 427.",
                        "Lag-2 and lag-12 are simple, no-training forecasting baselines.",
                        "No Naver values for these 566 keyword groups were supplied or fabricated.",
                        "2025 Sep-Dec is reported separately because source quality is under review."]}
    (args.out / "pilot_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
