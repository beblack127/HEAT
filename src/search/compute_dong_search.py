"""묶음별 월별 검색 지수 -> 행정동·칸별 검색 신호.

1. 검색량 제외 규칙(사전 고정, 2022년만 사용): 2022년 월평균이 기준 묶음의 1% 미만(지수 < 1)이거나 0인 달이 6개월 이상이면 제외
2. 동·칸마다 usable == 1 이고 제외되지 않은 이름의 가중치를 다시 합 1로 나눠 가중 평균 -> 칸 지수
3. 전체 = 있는 칸들의 평균
4. 변화(모두 서울 전체 대비 비중 기준): chg_1m 지난달보다, chg_3m 3개월 전보다(3개월 평균끼리),
   chg_3m_sa 3개월 전보다·계절 보정(2022~2024 달별 패턴 제거), chg_12m 1년 전보다(3개월 평균끼리)

사용: python3 compute_dong_search.py --series 검색량_묶음별_월별.csv --links 04_행정동_이름표_연결표.csv --out 결과폴더
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--series", required=True)
ap.add_argument("--links", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nonseoul", default="", help="서울 밖 시·군·구 이름 뿌리 목록 파일. 같은 뿌리의 묶음은 제외(예: 양평 -> 경기 양평군)")
ap.add_argument("--stoplist", default="", help="일반명사이거나 서울 밖 유명 지명과 겹치는 뿌리 목록(팀 검토 대상)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

s = pd.read_csv(a.series, encoding="utf-8-sig")
lk = pd.read_csv(a.links, encoding="utf-8-sig", dtype={"dong_code8": str})

# 1. 제외 규칙
y22 = s[s.month.str.startswith("2022")]
q = y22.groupby("keyword_group_id")["index"].agg(mean_2022="mean", zero_months_2022=lambda v: int((v <= 0).sum()))
q = q.reset_index().merge(s[["keyword_group_id", "root"]].drop_duplicates(), on="keyword_group_id", how="left")
ns = set(open(a.nonseoul, encoding="utf-8").read().split()) if a.nonseoul else set()
sl = set(open(a.stoplist, encoding="utf-8").read().split()) if a.stoplist else set()
q["nonseoul"] = q.root.isin(ns); q["stop"] = q.root.isin(sl)
q["excluded"] = ((q.mean_2022 < 1.0) | (q.zero_months_2022 >= 6) | q.nonseoul | q.stop).astype(int)
q["reason"] = np.select([q.nonseoul, q.stop, q.mean_2022 < 1.0, q.zero_months_2022 >= 6],
                        ["서울 밖 시·군 이름과 같음", "일반명사·서울 밖 유명 지명(검토 목록)", "2022 월평균 < 기준의 1%", "2022년 0인 달 6개월 이상"], "")
q = q.drop(columns=["nonseoul", "stop"])
q.round(4).to_csv(f"{a.out}/검색어묶음_품질.csv", index=False, encoding="utf-8-sig")

# 2. 동·칸 지수
L = lk[lk.usable == 1][["dong_code8", "tier", "keyword_group_id", "weight"]].copy()
L = L.merge(q[["keyword_group_id", "excluded"]], on="keyword_group_id", how="left")
L = L[L.excluded == 0].copy()
L["w"] = L.weight / L.groupby(["dong_code8", "tier"]).weight.transform("sum")
L = L[L.w.notna() & (L.w > 0)]
piv = s.pivot_table(index="keyword_group_id", columns="month", values="index")
months = list(piv.columns)
rows = []
for (c, t), g in L.groupby(["dong_code8", "tier"]):
    m = piv.loc[g.keyword_group_id].values  # 이름 x 월
    v = (g.w.values[:, None] * m).sum(axis=0)
    rows.append(pd.DataFrame({"dong_code8": c, "tier": t, "month": months, "index": v}))
ds = pd.concat(rows, ignore_index=True)
# 3. 전체(있는 칸 평균)
tot = ds.groupby(["dong_code8", "month"])["index"].mean().reset_index().assign(tier="전체")
ds = pd.concat([ds, tot], ignore_index=True).sort_values(["dong_code8", "tier", "month"])
# 4. 변화율
ds["ma3"] = ds.groupby(["dong_code8", "tier"])["index"].transform(lambda v: v.rolling(3).mean())
ds["yoy_ma3"] = ds.groupby(["dong_code8", "tier"])["ma3"].transform(lambda v: v / v.shift(12) - 1)
ds.loc[~np.isfinite(ds.yoy_ma3), "yoy_ma3"] = np.nan
# 서울 전체 흐름을 뺀 상대 지표: 그 달 서울 합계 대비 비중과, 그 비중의 3개월 평균 전년 대비 변화
ds["seoul_share"] = ds["index"] / ds.groupby(["tier", "month"])["index"].transform("sum")
ds["share_ma3"] = ds.groupby(["dong_code8", "tier"])["seoul_share"].transform(lambda v: v.rolling(3).mean())
ds["yoy_share_ma3"] = ds.groupby(["dong_code8", "tier"])["share_ma3"].transform(lambda v: v / v.shift(12) - 1)
ds.loc[~np.isfinite(ds.yoy_share_ma3), "yoy_share_ma3"] = np.nan
# 계절 계수: 동·종류마다 2022~2024년 '서울 대비 비중 / 12개월 중심 이동평균'의 달별 평균 (평균 1로 맞춤)
def seasonal(g):
    v = g.set_index("month")["seoul_share"]
    tr = v.rolling(12, min_periods=12).mean().shift(-6)   # i-5 ~ i+6 의 12개월 평균 (지도와 같은 정의)
    r = (v / tr)[(v.index < "2025-01") & tr.notna()]
    sf = r.groupby(r.index.str[5:]).mean().reindex([f"{m:02d}" for m in range(1, 13)])
    sf = sf / sf.mean() if sf.notna().all() and sf.mean() > 0 else pd.Series(1.0, index=sf.index)
    return pd.Series(g.month.str[5:].map(sf).values, index=g.index)
ds["seasonal_factor"] = ds.groupby(["dong_code8", "tier"], group_keys=False).apply(seasonal)
ds["share_sa"] = ds.seoul_share / ds.seasonal_factor
G = ds.groupby(["dong_code8", "tier"])
sa_ma3 = G["share_sa"].transform(lambda v: v.rolling(3).mean())
ds["chg_1m"] = G["seoul_share"].transform(lambda v: v / v.shift(1) - 1)          # 지난달보다
ds["chg_3m"] = ds.share_ma3 / G["share_ma3"].shift(3) - 1                          # 3개월 전보다 (3개월 평균끼리)
ds["chg_3m_sa"] = sa_ma3 / sa_ma3.groupby([ds.dong_code8, ds.tier]).shift(3) - 1   # 3개월 전보다, 계절 보정
ds["chg_12m"] = ds.yoy_share_ma3                                                   # 1년 전보다 (3개월 평균끼리)
for c in ("chg_1m", "chg_3m", "chg_3m_sa", "chg_12m"):
    ds.loc[~np.isfinite(ds[c]), c] = np.nan
ds.round(7).to_csv(f"{a.out}/행정동_검색지수_월별.csv", index=False, encoding="utf-8-sig")

n_dong = lk.dong_code8.nunique()
cov = ds[ds.tier != "전체"].groupby("tier").dong_code8.nunique().to_dict()
print(f"묶음 {len(q)}개 중 제외 {int(q.excluded.sum())}개 | 칸별 지수가 있는 동: {cov} / {n_dong} | 월 {months[0]}~{months[-1]}")
