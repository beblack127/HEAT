"""Download and aggregate official Seoul monthly mobility/living ZIPs for all dongs.

Run: python build_full_427_sources.py --repo PATH_TO_REPO --out WORK_DIR
Intermediate monthly aggregates are resumable and do not copy the raw ZIPs into out.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def official_codes(path: Path, month: str) -> set[str]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(f"ADMI_{month}.csv"))
        rows = csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig")))
        return {r["ADMI_CD"] for r in rows if r["SIDO_NM"] == "서울특별시"}


def aggregate_mobility(path: Path, month: str, codes: set[str]) -> list[dict]:
    sums = defaultdict(lambda: defaultdict(float))
    records = defaultdict(int)
    dates = set()
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with z.open(member) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
                if not {"d_admdong_cd", "move_purpose", "total_cnt", "etl_ymd"} <= set(reader.fieldnames or []):
                    raise ValueError(f"Missing mobility columns in {month}")
                for row in reader:
                    code = row["d_admdong_cd"]
                    if code not in codes:
                        continue
                    dates.add(row["etl_ymd"])
                    try:
                        value = float(row["total_cnt"])
                    except (TypeError, ValueError):
                        continue
                    sums[code][row["move_purpose"]] += value
                    records[code] += 1
    if len(dates) < 27 or len(sums) < 420:
        raise ValueError(f"Incomplete mobility {month}: {len(dates)} dates, {len(sums)} dongs")
    output = []
    for code in sorted(sums):
        purposes = sums[code]
        row = {"month": month[:4] + "-" + month[4:], "admin_dong_code": code,
               "days": len(dates), "records": records[code]}
        row.update({f"purpose_{p}_daily": purposes[str(p)] / len(dates) for p in range(1, 8)})
        row["target_457_daily"] = sum(purposes[str(p)] for p in (4, 5, 7)) / len(dates)
        row["all_arrivals_daily"] = sum(purposes.values()) / len(dates)
        output.append(row)
    return output


def aggregate_living(path: Path, month: str, codes: set[str]) -> list[dict]:
    totals = defaultdict(lambda: [0.0, 0, 0.0, 0])
    dates = set()
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        for member in members:
            with z.open(member) as f:
                encoding = "utf-8-sig" if f.peek(3)[:3] == b"\xef\xbb\xbf" else "cp949"
                reader = csv.reader(io.TextIOWrapper(f, encoding=encoding, newline=""))
                header = next(reader)
                if header[1:4] != ["시간대구분", "행정동코드", "총생활인구수"]:
                    raise ValueError(f"Unexpected living columns in {month}")
                for row in reader:
                    if len(row) < 4 or row[2] not in codes:
                        continue
                    try:
                        hour, value = int(row[1]), float(row[3])
                    except (TypeError, ValueError):
                        continue
                    dates.add(row[0])
                    acc = totals[row[2]]
                    acc[0] += value; acc[1] += 1
                    if 11 <= hour <= 21:
                        acc[2] += value; acc[3] += 1
    if len(dates) < 27 or len(totals) < 420:
        raise ValueError(f"Incomplete living {month}: {len(dates)} dates, {len(totals)} dongs")
    return [{"month": month[:4] + "-" + month[4:], "admin_dong_code": code,
             "living_all_hour_mean": acc[0] / acc[1],
             "living_day_mean": acc[2] / acc[3] if acc[3] else "",
             "hour_records": acc[1], "days": len(dates)}
            for code, acc in sorted(totals.items())]


def write_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import collect_context_population_mobility as source
    code_zip = args.repo / "data/reference/seoul_admin_dong_codes_official.zip"
    months = [f"{year}{m:02d}" for year in (2023, 2024, 2025) for m in range(1, 13)]
    codes_by_month = {month: official_codes(code_zip, month) for month in months}
    jobs = [(kind, month) for kind in ("mobility", "living") for month in months]

    def worker(kind: str, month: str) -> dict:
        dest = args.out / f"{kind}_{month}.csv"
        if dest.exists():
            with dest.open(encoding="utf-8-sig", newline="") as f:
                n = sum(1 for _ in csv.DictReader(f))
            if n >= 420:
                return {"kind": kind, "month": month, "dongs": n, "cached": True}
        temp_raw = Path(tempfile.gettempdir()) / f"codex_seoul_{kind}_{month}.zip"
        if temp_raw.exists() and not zipfile.is_zipfile(temp_raw):
            temp_raw.unlink()
        raw = source.download(kind, month)
        rows = (aggregate_mobility if kind == "mobility" else aggregate_living)(raw, month, codes_by_month[month])
        write_csv(dest, rows)
        return {"kind": kind, "month": month, "dongs": len(rows),
                "expected_official_codes": len(codes_by_month[month]),
                "source_bytes": raw.stat().st_size, "cached": False}

    manifest = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            manifest.append(result)
            print(result["kind"], result["month"], result["dongs"], "cached" if result["cached"] else "downloaded", flush=True)
            (args.out / "source_manifest.json").write_text(json.dumps({
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": sorted(manifest, key=lambda r: (r["kind"], r["month"]))}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
