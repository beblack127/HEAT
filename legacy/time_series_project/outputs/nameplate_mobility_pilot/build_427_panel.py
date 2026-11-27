"""Join all 427 nameplate dongs to 2023-2025 monthly source aggregates.

python build_427_panel.py --repo PATH_TO_REPO --nameplate PATH_TO_ZIP
                          --source WORK_DIR --out OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import pandas as pd


def zip_table(archive: zipfile.ZipFile, prefix: str) -> pd.DataFrame:
    names = [n for n in archive.namelist() if n.split("/")[-1].startswith(prefix)
             and n.endswith(".csv") and not n.startswith("__MACOSX/") and "/._" not in n]
    if len(names) != 1:
        raise ValueError(f"Expected one ZIP CSV starting with {prefix}")
    return pd.read_csv(io.BytesIO(archive.read(names[0])), encoding="utf-8-sig",
                       dtype={"dong_code8": str, "dong_code10": str, "keyword_group_id": str})


def read_source(folder: Path, kind: str) -> pd.DataFrame:
    files = sorted(folder.glob(f"{kind}_20????.csv"))
    if len(files) != 36:
        raise ValueError(f"Expected 36 monthly {kind} aggregates, got {len(files)}")
    result = pd.concat((pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                        for p in files), ignore_index=True)
    if result.duplicated(["admin_dong_code", "month"]).any():
        raise ValueError(f"Duplicate {kind} code-month")
    return result


def official_sets(path: Path) -> dict[str, set[str]]:
    result = {}
    with zipfile.ZipFile(path) as z:
        for year in (2023, 2024, 2025):
            for m in range(1, 13):
                period = f"{year}{m:02d}"
                name = next(n for n in z.namelist() if n.endswith(f"ADMI_{period}.csv"))
                rows = csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig")))
                result[f"{year}-{m:02d}"] = {r["ADMI_CD"] for r in rows if r["SIDO_NM"] == "서울특별시"}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--nameplate", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.nameplate) as archive:
        dongs = zip_table(archive, "01_")
        groups = zip_table(archive, "02_")
        links = zip_table(archive, "04_")
    if len(dongs) != 427 or dongs.dong_code8.nunique() != 427:
        raise ValueError("Nameplate must have 427 unique dongs")
    links = links.merge(groups[["keyword_group_id", "root", "keywords"]],
                        on="keyword_group_id", how="left", validate="many_to_one")
    links.to_csv(args.out / "nameplate_links_427.csv", index=False, encoding="utf-8-sig")

    compact = []
    for _, dong in dongs.iterrows():
        code = dong["dong_code8"]
        row = {"admin_dong_code": code, "gu": dong["gu"], "dong_name": dong["dong_name"],
               "land_km2": dong["land_km2"]}
        for tier in ("동네", "역", "권역"):
            selected = links[(links.dong_code8 == code) & (links.tier == tier) & (links.usable == 1)]
            row[f"n_{tier}"] = len(selected)
            row[f"names_{tier}"] = " | ".join(f"{r['name']} ({float(r['weight']):.4f})"
                                             for _, r in selected.iterrows())
            row[f"keyword_groups_{tier}"] = " | ".join(selected.keyword_group_id.unique())
        compact.append(row)
    compact = pd.DataFrame(compact).sort_values("admin_dong_code")
    compact.to_csv(args.out / "nameplate_admin_dong_427.csv", index=False, encoding="utf-8-sig")

    mob = read_source(args.source, "mobility")
    living = read_source(args.source, "living")
    retired = mob[~mob.admin_dong_code.isin(compact.admin_dong_code)].copy()
    retired.to_csv(args.out / "retired_code_mobility_history.csv", index=False, encoding="utf-8-sig")
    if not set(mob.admin_dong_code).issuperset(set(compact.admin_dong_code) - {"11230515", "11230533"}):
        raise ValueError("Some stable nameplate codes absent from mobility source")
    months = sorted(official_sets(args.repo / "data/reference/seoul_admin_dong_codes_official.zip"))
    codes = compact.admin_dong_code.tolist()
    panel = pd.MultiIndex.from_product([codes, months], names=["admin_dong_code", "month"]).to_frame(index=False)
    panel = panel.merge(compact, on="admin_dong_code", how="left", validate="many_to_one")
    panel = panel.merge(mob, on=["admin_dong_code", "month"], how="left", validate="one_to_one", suffixes=("", "_mob"))
    panel = panel.merge(living[["admin_dong_code", "month", "living_all_hour_mean", "living_day_mean",
                                "hour_records"]], on=["admin_dong_code", "month"], how="left", validate="one_to_one")
    official = official_sets(args.repo / "data/reference/seoul_admin_dong_codes_official.zip")
    panel["historical_status"] = ["official" if code in official[month] else "not_yet_created"
                                  for code, month in zip(panel.admin_dong_code, panel.month)]
    panel["data_status"] = ["observed" if pd.notna(v) else ("not_yet_created" if status == "not_yet_created" else "missing_source")
                            for v, status in zip(panel.target_457_daily, panel.historical_status)]
    panel = panel.sort_values(["admin_dong_code", "month"])
    if len(panel) != 427 * 36 or panel.duplicated(["admin_dong_code", "month"]).any():
        raise ValueError("Invalid final panel")
    panel.to_csv(args.out / "admin_dong_month_panel_427_2023_2025.csv", index=False, encoding="utf-8-sig")
    status = panel.groupby("admin_dong_code").data_status.value_counts().unstack(fill_value=0)
    result = {"nameplate_dongs": len(compact), "links": len(links), "panel_rows": len(panel),
              "observed_rows": int((panel.data_status == "observed").sum()),
              "not_yet_created_rows": int((panel.data_status == "not_yet_created").sum()),
              "missing_source_rows": int((panel.data_status == "missing_source").sum()),
              "fully_observed_dongs": int((status.get("observed", 0) == 36).sum()),
              "retired_codes_outside_nameplate": sorted(retired.admin_dong_code.unique().tolist()),
              "dongs_with_incomplete_history": status[status.get("observed", 0) < 36].to_dict("index")}
    (args.out / "full_427_panel_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
