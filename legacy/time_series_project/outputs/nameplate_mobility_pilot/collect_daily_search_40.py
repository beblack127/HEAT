"""Collect daily NAVER API HUB search trends for the existing 40-dong pilot.

Uses the HEAT nameplate links and quality exclusions. Responses are cached per
batch without credentials so interrupted runs can resume safely.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEAT = ROOT / "work" / "HEAT"
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "daily_search_40"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HEAT / "src" / "search"))
from collect_datalab import call_api, read_env  # noqa: E402

START, END = "2023-01-01", "2025-08-31"
DATES = pd.date_range(START, END, freq="D").strftime("%Y-%m-%d").tolist()


def inputs():
    b = HEAT / "data" / "processed"
    pilot = pd.read_csv(ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_pilot" / "aligned_panel_40.csv",
                        dtype={"admin_dong_code": str})
    codes = set(pilot.admin_dong_code)
    groups = pd.read_csv(b / "nametags" / "02_검색어_묶음.csv", encoding="utf-8-sig")
    links = pd.read_csv(b / "nametags" / "04_행정동_이름표_연결표.csv", encoding="utf-8-sig", dtype={"dong_code8": str})
    quality = pd.read_csv(b / "search" / "검색어묶음_품질.csv", encoding="utf-8-sig")
    links = links.loc[links.dong_code8.isin(codes) & links.usable.eq(1)].merge(
        quality[["keyword_group_id", "excluded"]], on="keyword_group_id", validate="many_to_one")
    links = links.loc[links.excluded.eq(0)].copy()
    if links.dong_code8.nunique() != 40:
        raise ValueError("Search nameplates do not cover all pilot dongs")
    groups = groups.loc[groups.keyword_group_id.isin(links.keyword_group_id)].sort_values("keyword_group_id")
    units = []
    for row in groups.itertuples():
        keywords = [x.strip() for x in row.keywords.split("|") if x.strip()]
        for j in range(0, len(keywords), 5):
            units.append((f"{row.keyword_group_id}#{j // 5}", row.keyword_group_id, keywords[j:j + 5]))
    anchor_row = pd.read_csv(b / "nametags" / "02_검색어_묶음.csv", encoding="utf-8-sig")
    anchor_row = anchor_row.loc[anchor_row.root.eq("문래")].iloc[0]
    anchor = [x.strip() for x in anchor_row.keywords.split("|") if x.strip()][:5]
    return codes, links, units, anchor


def series(result: dict, title: str) -> np.ndarray:
    entries = next(r["data"] for r in result["results"] if r["title"] == title)
    values = {x["period"]: float(x["ratio"]) for x in entries}
    return np.array([values.get(d, 0.0) for d in DATES], dtype=float)


def main() -> None:
    global OUT, RAW
    parser = argparse.ArgumentParser()
    parser.add_argument("--ages-19to39", action="store_true", help="NAVER age buckets 19-24,25-29,30-34,35-39")
    args = parser.parse_args()
    if args.ages_19to39:
        OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "daily_search_19to39_40"
        RAW = OUT / "raw"
        RAW.mkdir(parents=True, exist_ok=True)
    codes, links, units, anchor = inputs()
    batches = [units[i:i + 4] for i in range(0, len(units), 4)]
    credentials = None
    for i, batch in enumerate(batches):
        request = {"startDate": START, "endDate": END, "timeUnit": "date",
                   "keywordGroups": [{"groupName": "ANCHOR", "keywords": anchor}] +
                                    [{"groupName": unit, "keywords": words} for unit, _, words in batch]}
        if args.ages_19to39:
            request["ages"] = ["3", "4", "5", "6"]
        target = RAW / f"batch_{i:04d}.json"
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["request"] != request:
                raise ValueError(f"Cached request differs: {target}")
        else:
            if credentials is None:
                credentials = read_env(str(HEAT / ".env"))
            response = call_api(request, credentials)
            if not response.get("results") or len(response["results"]) != len(request["keywordGroups"]):
                raise ValueError(f"Unexpected response for batch {i}")
            target.write_text(json.dumps({"request": request, "response": response}, ensure_ascii=False), encoding="utf-8")
            time.sleep(.2)
        if (i + 1) % 10 == 0 or i + 1 == len(batches):
            print(f"daily search batches: {i + 1}/{len(batches)}", flush=True)

    reference = json.loads((RAW / "batch_0000.json").read_text(encoding="utf-8"))["response"]
    ref = series(reference, "ANCHOR")
    base = ref[:365].mean()
    if base <= 0:
        raise ValueError("Nonpositive anchor baseline")
    parts = []
    anchor_correlations = []
    for i, batch in enumerate(batches):
        response = json.loads((RAW / f"batch_{i:04d}.json").read_text(encoding="utf-8"))["response"]
        other_anchor = series(response, "ANCHOR")
        # Calibrate only on 2023, which is already known at all 2025 origins.
        scale = ref[:365].sum() / other_anchor[:365].sum()
        anchor_correlations.append(float(np.corrcoef(ref, other_anchor)[0, 1]))
        for unit, group_id, _ in batch:
            v = series(response, unit) * scale / base * 100
            parts.append(pd.DataFrame({"keyword_group_id": group_id, "date": DATES, "index": v}))
    group_daily = pd.concat(parts, ignore_index=True).groupby(["keyword_group_id", "date"], as_index=False).index.sum()
    group_daily.to_csv(OUT / "group_daily.csv", index=False, encoding="utf-8-sig")
    links["w"] = links.weight / links.groupby(["dong_code8", "tier"]).weight.transform("sum")
    attached = links[["dong_code8", "tier", "keyword_group_id", "w"]].merge(group_daily, on="keyword_group_id", validate="many_to_many")
    attached["weighted_index"] = attached.w * attached["index"]
    tiers = attached.groupby(["dong_code8", "tier", "date"], as_index=False).weighted_index.sum()
    dong = tiers.groupby(["dong_code8", "date"], as_index=False).weighted_index.mean()
    dong = dong.rename(columns={"dong_code8": "admin_dong_code", "weighted_index": "search_index"})
    if len(dong) != len(codes) * len(DATES):
        raise ValueError("Incomplete dong-day search coverage")
    dong.to_csv(OUT / "dong_daily.csv", index=False, encoding="utf-8-sig")
    audit = {"dates": len(DATES), "dongs": len(codes), "groups": len(set(u[1] for u in units)),
             "units": len(units), "batches": len(batches),
             "min_anchor_shape_correlation": min(anchor_correlations),
             "search_index_definition": "relative to anchor mean in 2023; not actual query count or Seoul share",
             "age_filter": "19-39 (NAVER buckets 3-6), includes age 19; target mobility is 20-39" if args.ages_19to39 else "none; matches existing monthly all-age search"}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
