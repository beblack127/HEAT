"""External construct check of the 4+5+7 mobility target.

Joins published dong-level quarterly footfall directly by administrative code.
Food-service transaction counts are available here by commercial area and are
allocated to dongs with HEAT's polygon-overlap shares. Both are estimated
commercial indicators, not direct ground truth for dong mobility.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "work" / "Time-series-Project"
HEAT = ROOT / "work" / "HEAT"
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "target_validation"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN_QUARTERS = ["20231", "20232", "20233", "20234", "20241", "20242", "20251", "20252"]


def rho(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 5 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def load_mobility() -> pd.DataFrame:
    folder = REPO / "data" / "context_monthly"
    months = pd.concat([
        pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
        for p in sorted(folder.glob("mobility_20????.csv"))
    ], ignore_index=True)
    assert months.admin_dong_code.nunique() == 40
    assert not months.duplicated(["admin_dong_code", "month"]).any()
    months["quarter"] = months.month.str[:4] + ((months.month.str[5:7].astype(int)-1)//3+1).astype(str)
    # Source monthly values are daily averages. Weight by days to recover each
    # quarter's daily average without giving February the same duration as March.
    months["days"] = pd.PeriodIndex(months.month, freq="M").days_in_month
    for c in ("mobility_shop_tour_other_daily", "mobility_all_arrivals_daily", "mobility_shop_tour_daily"):
        months[c + "_weighted"] = months[c] * months.days
    result = months.groupby(["admin_dong_code", "quarter"], as_index=False).agg(
        days=("days", "sum"),
        target_sum=("mobility_shop_tour_other_daily_weighted", "sum"),
        all_sum=("mobility_all_arrivals_daily_weighted", "sum"),
        shop_tour_sum=("mobility_shop_tour_daily_weighted", "sum"),
    )
    for source, target in (("target_sum", "mobility_457_daily"), ("all_sum", "mobility_all_daily"),
                           ("shop_tour_sum", "mobility_45_daily")):
        result[target] = result[source] / result.days
    return result[["admin_dong_code", "quarter", "mobility_457_daily", "mobility_all_daily", "mobility_45_daily"]]


def load_bridge(codes: set[str]) -> pd.DataFrame:
    path = HEAT / "data" / "processed" / "nametags" / "06_행정동_상권_연결표.csv"
    bridge = pd.read_csv(path, encoding="utf-8-sig", dtype={"sangkwon_code": str, "dong_code8": str})
    bridge = bridge.rename(columns={"sangkwon_code": "area_code", "dong_code8": "admin_dong_code",
                                    "sangkwon_share": "weight"})
    bridge = bridge.loc[bridge.admin_dong_code.isin(codes), ["area_code", "admin_dong_code", "weight", "overlap_m2"]]
    if bridge.duplicated(["area_code", "admin_dong_code"]).any() or bridge.weight.le(0).any():
        raise ValueError("Invalid commercial-area bridge")
    return bridge


def load_footfall() -> pd.DataFrame:
    folder = OUT / "direct_sources"
    parts = []
    for q in CLEAN_QUARTERS:
        p = folder / f"footfall_dong_{q}.csv"
        t = pd.read_csv(p, encoding="cp949", dtype={"행정동_코드": str},
                        usecols=["기준_년분기_코드", "행정동_코드", "총_유동인구_수"])
        t = t.rename(columns={"기준_년분기_코드": "quarter", "행정동_코드": "admin_dong_code",
                              "총_유동인구_수": "footfall"})
        t.quarter = t.quarter.astype(str)
        parts.append(t)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["admin_dong_code", "quarter"]).any():
        raise ValueError("Duplicate dong-quarter footfall")
    return result


def load_food_sales() -> pd.DataFrame:
    parts = []
    for year in (2023, 2024, 2025):
        path = REPO / "data" / "reference" / f"seoul_commercial_sales_{year}.zip"
        with zipfile.ZipFile(path) as z:
            names = [x.filename for x in z.infolist() if x.filename.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one sales CSV in {path}")
            with z.open(names[0]) as f:
                t = pd.read_csv(f, encoding="cp949", dtype={"상권_코드": str, "서비스_업종_코드": str},
                                usecols=["기준_년분기_코드", "상권_코드", "서비스_업종_코드", "당월_매출_건수"])
        t = t.loc[t["서비스_업종_코드"].str.startswith("CS100", na=False)]
        t = t.rename(columns={"기준_년분기_코드": "quarter", "상권_코드": "area_code",
                              "당월_매출_건수": "food_transactions"})
        t.quarter = t.quarter.astype(str)
        t = t.loc[t.quarter.isin(CLEAN_QUARTERS)]
        parts.append(t.groupby(["quarter", "area_code"], as_index=False).food_transactions.sum())
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["area_code", "quarter"]).any():
        raise ValueError("Duplicate area-quarter sales")
    return result


def allocate(area: pd.DataFrame, bridge: pd.DataFrame, value: str) -> pd.DataFrame:
    joined = area.merge(bridge, on="area_code", how="inner", validate="many_to_many")
    joined["allocated"] = joined[value] * joined.weight
    return joined.groupby(["admin_dong_code", "quarter"], as_index=False).agg(
        **{value: ("allocated", "sum")}, areas=("area_code", "nunique")
    )


def add_yoy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["admin_dong_code", "quarter"]).copy()
    for col in ("mobility_457_daily", "mobility_all_daily", "mobility_45_daily", "footfall", "food_transactions"):
        prior = df[["admin_dong_code", "quarter", col]].copy()
        prior.quarter = (prior.quarter.str[:4].astype(int)+1).astype(str) + prior.quarter.str[-1]
        prior = prior.rename(columns={col: col+"_prior"})
        df = df.merge(prior, on=["admin_dong_code", "quarter"], how="left", validate="one_to_one")
        df[col+"_yoy_log"] = np.log1p(df[col]) - np.log1p(df[col+"_prior"])
    return df


def main() -> None:
    mobility = load_mobility()
    codes = set(mobility.admin_dong_code)
    bridge = load_bridge(codes)
    footfall = load_footfall()
    sales = load_food_sales()
    panel = mobility.merge(footfall, on=["admin_dong_code", "quarter"], how="left", validate="one_to_one")
    panel = panel.merge(allocate(sales, bridge, "food_transactions"),
                        on=["admin_dong_code", "quarter"], how="left", validate="one_to_one")
    panel = panel.loc[panel.quarter.isin(CLEAN_QUARTERS)].copy()
    panel = add_yoy(panel)
    panel.to_csv(OUT / "aligned_quarterly_40.csv", index=False, encoding="utf-8-sig")

    # Cross-sectional level correlation is secondary because commercial areas
    # cover only part of an administrative dong and the mapping is area based.
    levels = {}
    for outcome in ("footfall", "food_transactions"):
        levels[outcome] = {}
        for signal in ("mobility_457_daily", "mobility_all_daily", "mobility_45_daily"):
            vals = [rho(g[signal], g[outcome]) for _, g in panel.groupby("quarter")]
            levels[outcome][signal] = {"median_quarter_spearman": round(float(np.nanmedian(vals)), 3),
                                       "by_quarter": dict(zip(sorted(panel.quarter.unique()), [round(x, 3) for x in vals]))}

    # Main analysis: matched year-on-year change in non-anomalous quarters.
    # De-mean each quarter to remove common citywide movement before pooling.
    change = panel.loc[panel.quarter.isin(["20241", "20242", "20251", "20252"])].copy()
    change_cols = [c for c in change.columns if c.endswith("_yoy_log")]
    for col in change_cols:
        change[col+"_demeaned"] = change[col] - change.groupby("quarter")[col].transform("mean")
    change.to_csv(OUT / "yoy_change_40.csv", index=False, encoding="utf-8-sig")
    changes = {}
    for outcome in ("footfall", "food_transactions"):
        changes[outcome] = {}
        b = outcome+"_yoy_log_demeaned"
        for signal in ("mobility_457_daily", "mobility_all_daily", "mobility_45_daily"):
            a = signal+"_yoy_log_demeaned"
            valid = change[a].notna() & change[b].notna()
            changes[outcome][signal] = {"n_dong_quarters": int(valid.sum()),
                                        "pooled_spearman": round(rho(change[a], change[b]), 3),
                                        "by_quarter": {q: round(rho(g[a], g[b]), 3)
                                                       for q, g in change.groupby("quarter")}}
    coverage = {"mobility_dongs": len(codes), "bridge_dongs": bridge.admin_dong_code.nunique(),
                "bridge_areas": bridge.area_code.nunique(),
                "panel_rows": len(panel), "footfall_rows": int(panel.footfall.notna().sum()),
                "sales_rows": int(panel.food_transactions.notna().sum()),
                "areas_per_dong_median_for_sales": float(panel.groupby("admin_dong_code").areas.median().median()),
                "clean_quarters": CLEAN_QUARTERS,
                "excluded_quarters": ["20243", "20244", "20253", "20254"]}
    result = {"coverage": coverage, "level_spearman_secondary": levels,
              "yoy_change_spearman_primary": changes,
              "method_note": "Footfall is official dong-level quarterly data joined by code. Food transactions are commercial-area estimates allocated by polygon overlap. Same-dong year-over-year change is primary; no causal interpretation."}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": coverage, "yoy_change_spearman_primary": changes,
                      "level_median": {k: {s: v["median_quarter_spearman"] for s, v in d.items()}
                                       for k, d in levels.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
