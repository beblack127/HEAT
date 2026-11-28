"""Descriptive per-dong search/mobility correlations on the 40-dong pilot."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "search_correlation"
OUT.mkdir(parents=True, exist_ok=True)
MOB = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly"
SEARCH = ROOT / "work" / "HEAT" / "data" / "processed" / "search" / "행정동_검색지수_월별.csv"
NAMES = ROOT / "work" / "HEAT" / "data" / "processed" / "nametags" / "01_행정동_목록.csv"
END = pd.Period("2025-08", freq="M")
BAD = {pd.Period(x, freq="M") for x in ("2024-08", "2024-09", "2024-10")}


def corr(a, b):
    valid = a.notna() & b.notna()
    aa, bb = a[valid], b[valid]
    if len(aa) < 6 or aa.nunique() < 3 or bb.nunique() < 3:
        return float("nan"), int(len(aa))
    return float(spearmanr(aa, bb).statistic), int(len(aa))


mob = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                 for p in sorted(MOB.glob("mobility_20????.csv"))], ignore_index=True)
mob["period"] = pd.PeriodIndex(mob.month, freq="M")
mob = mob.loc[mob.period.le(END)].copy()
assert mob.admin_dong_code.nunique() == 40
assert not mob.duplicated(["admin_dong_code", "period"]).any()
search = pd.read_csv(SEARCH, encoding="utf-8-sig", dtype={"dong_code8": str})
search = search.loc[search.tier.eq("전체"), ["dong_code8", "month", "seoul_share", "index"]].copy()
search = search.rename(columns={"dong_code8": "admin_dong_code", "seoul_share": "search_share", "index": "search_index"})
search["period"] = pd.PeriodIndex(search.month, freq="M")
assert not search.duplicated(["admin_dong_code", "period"]).any()
names = pd.read_csv(NAMES, encoding="utf-8-sig", dtype={"dong_code8": str})
names = names.set_index("dong_code8")[["gu", "dong_name"]]

panel = mob[["admin_dong_code", "period", "mobility_shop_tour_other_daily", "mobility_all_arrivals_daily"]].copy()
panel = panel.rename(columns={"mobility_shop_tour_other_daily": "mobility_457", "mobility_all_arrivals_daily": "mobility_all"})
for lag in (0, 1, 2):
    part = search[["admin_dong_code", "period", "search_share", "search_index"]].copy()
    part["period"] += lag
    part = part.rename(columns={"search_share": f"search_share_lag{lag}", "search_index": f"search_index_lag{lag}"})
    panel = panel.merge(part, on=["admin_dong_code", "period"], how="left", validate="one_to_one")

for lag in (0, 2):
    part = search[["admin_dong_code", "period", "search_share"]].copy()
    prior = part.copy()
    prior["period"] += 12
    prior = prior.rename(columns={"search_share": "search_prior"})
    part = part.merge(prior, on=["admin_dong_code", "period"], how="left", validate="one_to_one")
    part["search_yoy"] = np.log1p(1_000_000 * part.search_share) - np.log1p(1_000_000 * part.search_prior)
    part["period"] += lag
    panel = panel.merge(part[["admin_dong_code", "period", "search_yoy"]].rename(columns={"search_yoy": f"search_yoy_lag{lag}"}),
                        on=["admin_dong_code", "period"], how="left", validate="one_to_one")
prior = panel[["admin_dong_code", "period", "mobility_457"]].copy()
prior["period"] += 12
prior = prior.rename(columns={"mobility_457": "mobility_prior"})
panel = panel.merge(prior, on=["admin_dong_code", "period"], how="left", validate="one_to_one")
panel["mobility_yoy"] = np.log1p(panel.mobility_457) - np.log1p(panel.mobility_prior)
panel["month"] = panel.period.astype(str)
panel.drop(columns="period").to_csv(OUT / "aligned_40_dong_month.csv", index=False, encoding="utf-8-sig")

rows = []
for code, group in panel.groupby("admin_dong_code"):
    group = group.sort_values("period")
    row = {"admin_dong_code": code, "gu": names.loc[code, "gu"], "dong_name": names.loc[code, "dong_name"]}
    for metric, a, b, subset in (
        ("same_month_share", "search_share_lag0", "mobility_457", group),
        ("same_month_share_clean", "search_share_lag0", "mobility_457", group.loc[~group.period.isin(BAD)]),
        ("search_1m_ahead", "search_share_lag1", "mobility_457", group),
        ("search_2m_ahead", "search_share_lag2", "mobility_457", group),
        ("search_2m_ahead_clean", "search_share_lag2", "mobility_457", group.loc[~group.period.isin(BAD)]),
        ("same_month_index", "search_index_lag0", "mobility_457", group),
        ("yoy_same_month", "search_yoy_lag0", "mobility_yoy", group.loc[~group.period.isin(BAD) & ~group.period.isin({x + 12 for x in BAD})]),
        ("yoy_search_2m_ahead", "search_yoy_lag2", "mobility_yoy", group.loc[~group.period.isin(BAD) & ~group.period.isin({x + 12 for x in BAD})]),
    ):
        value, n = corr(subset[a], subset[b])
        row[metric + "_rho"] = round(value, 4) if np.isfinite(value) else np.nan
        row[metric + "_n"] = n
    rows.append(row)
result = pd.DataFrame(rows).sort_values("admin_dong_code")
result.to_csv(OUT / "per_dong_correlations.csv", index=False, encoding="utf-8-sig")
summary = {"period": "2023-01 to 2025-08", "dongs": len(result), "method": "within-dong Spearman; search share is Seoul-relative, not count",
           "yoy_exclusions": "2024-08 to 2024-10 anomalous mobility months and their 2025 matching months; 2025-09 onward excluded for source quality"}
for metric in ("same_month_share", "same_month_share_clean", "search_1m_ahead", "search_2m_ahead", "search_2m_ahead_clean", "same_month_index", "yoy_same_month", "yoy_search_2m_ahead"):
    x = result[metric + "_rho"].dropna()
    summary[metric] = {"median_rho": round(float(x.median()), 3), "mean_rho": round(float(x.mean()), 3),
                       "positive_dongs": int(x.gt(0).sum()), "negative_dongs": int(x.lt(0).sum()),
                       "at_least_0_3": int(x.ge(.3).sum()), "at_most_minus_0_3": int(x.le(-.3).sum()),
                       "n_min": int(result[metric + "_n"].min()), "n_max": int(result[metric + "_n"].max())}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# 네이버 검색 관심도와 생활이동의 동별 상관",
    "",
    "2026-11-14 계산. 40개 동, 2023-01~2025-08. 결과는 기술통계이며 인과·예측력 검정이 아니다.",
    "",
    "## 방법",
    "",
    "- 검색: HEAT `tier=전체`의 `seoul_share`(서울 대비 검색 비중, 검색 건수 아님). 이동: 목적 4+5+7의 도착 이동량 일평균.",
    "- 동별 월 시계열의 Spearman 순위상관. `같은 달`은 검색 m 대 이동 m, `검색 2개월 선행`은 검색 m-2 대 이동 m이다.",
    "- 주 표는 2024-08~10의 광범위한 생활이동 급증 월을 제외한 29개월. 전체 32개월·전년동월 변화 결과도 CSV에 보존했다.",
    "- 전년동월 로그 변화 비교는 2024-08~10 및 그에 대응하는 2025-08을 제외해 동별 16개월이다. 계절성·추세가 수준 상관에 주는 영향을 점검하려는 보조 분석이다.",
    "",
    "## 요약",
    "",
    f"- 같은 달: 동별 상관 중앙값 {summary['same_month_share_clean']['median_rho']:+.3f}; 양수 {summary['same_month_share_clean']['positive_dongs']}/40동.",
    f"- 검색 2개월 선행: 중앙값 {summary['search_2m_ahead_clean']['median_rho']:+.3f}; 양수 {summary['search_2m_ahead_clean']['positive_dongs']}/40동.",
    f"- 전년동월 변화, 검색 2개월 선행: 중앙값 {summary['yoy_search_2m_ahead']['median_rho']:+.3f}; 동별 16개월로 불안정하다.",
    "",
    "## 동별 결과 (2024-08~10 제외)",
    "",
    "| 자치구 | 행정동 | 같은 달 ρ | 검색 2개월 선행 ρ |",
    "|---|---|---:|---:|",
]
for r in result.itertuples(index=False):
    lines.append(f"| {r.gu} | {r.dong_name} | {r.same_month_share_clean_rho:+.2f} | {r.search_2m_ahead_clean_rho:+.2f} |")
lines += [
    "",
    "## 해석 범위",
    "",
    "같은 달의 동별 상관이 일부 높아도 검색이 이동을 선행한다는 뜻은 아니다. 검색 2개월 선행 관계는 전체적으로 거의 0이고 동별 방향이 크게 다르다. 각 동의 월 수가 적고 시계열 자기상관이 있으므로 이 수치를 독립 표본의 유의확률로 해석하지 않는다. 40개 동 파일럿을 서울 427개 동의 결과로 일반화하지 않는다.",
    "",
    "원자료: `work/HEAT/data/processed/search/행정동_검색지수_월별.csv`, `work/Time-series-Project/data/context_monthly/mobility_*.csv`. 재현: `analyze_dong_search_correlation.py`. 상세: `search_correlation/per_dong_correlations.csv`, `search_correlation/aligned_40_dong_month.csv`.",
]
(OUT / "REPORT_2026-11-14.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
