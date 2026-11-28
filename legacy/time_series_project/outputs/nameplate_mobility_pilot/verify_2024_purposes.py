"""Reaggregate purposes 1..7 from current official monthly ZIPs for 40 dongs.

python verify_2024_purposes.py --repo PATH_TO_REPO --out OUTPUT_DIR
Downloads 2024-07..11 via the repository's existing Seoul portal downloader.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import collect_context_population_mobility as source
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for month in ("202407", "202408", "202409", "202410", "202411"):
        path = source.download("mobility", month)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        totals = defaultdict(float)
        days = set()
        counts = defaultdict(int)
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with z.open(member) as f:
                    rows = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
                    for r in rows:
                        if r["d_admdong_cd"] not in source.dongs:
                            continue
                        purpose = r["move_purpose"]
                        totals[purpose] += float(r["total_cnt"])
                        counts[purpose] += 1
                        days.add(r["etl_ymd"])
        result = {"month": month[:4] + "-" + month[4:], "source_bytes": path.stat().st_size,
                  "source_sha256": digest.hexdigest(), "days": len(days),
                  "total_cnt_daily_by_purpose": {k: round(v / len(days), 2) for k, v in sorted(totals.items())},
                  "records_by_purpose": dict(sorted(counts.items()))}
        results.append(result)
        print(result["month"], result["total_cnt_daily_by_purpose"], flush=True)
    (args.out / "eda_2024_purpose_verification.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
