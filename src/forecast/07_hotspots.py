"""현재 핫스팟·떠오르는 곳 리스트 (기술 통계, 모델 선택에는 쓰지 않음).

검색 (2026-08~10)
  현재 핫스팟    최근 3개월 평균 서울 대비 비중
  떠오르는 곳    최근 3개월 평균 비중 / 1년 전 같은 3개월 평균 − 1  (비중 하위 25% 동 제외)
개폐업 (2025-10 ~ 2026-09, 일반·휴게음식점 합침, include == 1)
  현재 핫스팟    12개월 개업 수
  떠오르는 곳    영업 점포 증가율 S(2026-09) / S(2025-09) − 1  (점포 하위 25% 동 제외)
                 같은 날 10곳 이상 개업과 일괄 폐업이 함께 있는 동(재등록으로 보임)은 따로 표시하고 순위에서 뺀다
출력: reports/forecast/07_hotspots_*.csv, 07_hotspot_trends.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C, log_run, read_csv, set_korean_font

set_korean_font()
R = C.REPORTS
N = 10
S_NOW, S_CUR, S_PREV, S_HALF = "2026-10", ("2026-08", "2026-10"), ("2025-08", "2025-10"), ("2026-02", "2026-04")
F_WIN, F_PREV_WIN, F_END, F_START, F_PREV = ("2025-10", "2026-09"), ("2024-10", "2025-09"), "2026-09", "2025-09", "2024-09"
MASS_OPEN = 10

dl = read_csv(C.DONG_LIST_CSV, dtype={"dong_code8": str}).set_index("dong_code8")
name = lambda i: f"{dl.gu[i]} {dl.dong_name[i]}"  # noqa: E731

# ---------------- 검색 ----------------
s = pd.read_parquet(C.INTERIM / "search_monthly.parquet")
sh = s[s.tier == "전체"].pivot(index="month", columns="dong_code8", values="share").sort_index()
cur, prev, half = (sh.loc[a:b].mean() for a, b in (S_CUR, S_PREV, S_HALF))
yoy, hoh = cur / prev - 1, cur / half - 1
rank = cur.rank(ascending=False).astype(int)

hot_s = cur.sort_values(ascending=False).head(N)
hot_s = pd.DataFrame({"dong_code8": hot_s.index, "동": [name(k) for k in hot_s.index],
                      "서울대비비중_%": (hot_s * 100).round(2).values, "1년전대비_%": (yoy[hot_s.index] * 100).round(0).values})
elig = cur[cur.rank(pct=True) >= C.TOPK_SEARCH_MIN_LEVEL_PCT].index
em = yoy[elig].sort_values(ascending=False).head(N)
em_s = pd.DataFrame({"dong_code8": em.index, "동": [name(k) for k in em.index],
                     "1년전대비_%": (em * 100).round(0).values, "6개월전대비_%": (hoh[em.index] * 100).round(0).values,
                     "서울대비비중_%": (cur[em.index] * 100).round(3).values, "비중순위": rank[em.index].values})

# ---------------- 개폐업 ----------------
f = pd.read_parquet(C.INTERIM / "fnb_monthly.parquet")
f = f[f.level == "dong"]
W = lambda c: f.pivot(index="month", columns="series_id", values=c).sort_index()  # noqa: E731
O, Ck, S = W("O"), W("C_v_keep"), W("S")
o12, c12 = O.loc[F_WIN[0]:F_WIN[1]].sum(), Ck.loc[F_WIN[0]:F_WIN[1]].sum()
o12p = O.loc[F_PREV_WIN[0]:F_PREV_WIN[1]].sum()
s0, s1, sp = S.loc[F_START], S.loc[F_END], S.loc[F_PREV]
growth, growth_p = s1 / s0 - 1, s0 / sp - 1

st = pd.read_parquet(C.INTERIM / "stores_mapped.parquet")
w = (st.open_date >= f"{F_WIN[0]}-01") & (st.open_date <= pd.Period(F_WIN[1]).end_time)
so = st[w].groupby(["dong_code8", "open_date"]).size()
mass = so[so >= MASS_OPEN].groupby(level=0).sum()
wc = (st.close_date >= f"{F_WIN[0]}-01") & (st.close_date <= pd.Period(F_WIN[1]).end_time)
bulk = st[wc & st.bulk_closure].groupby("dong_code8").size()

hot_f = o12.sort_values(ascending=False).head(N)
hot_f = pd.DataFrame({"dong_code8": hot_f.index, "동": [name(k) for k in hot_f.index], "개업_12개월": hot_f.astype(int).values,
                      "폐업_12개월": c12[hot_f.index].astype(int).values, "영업점포": s1[hot_f.index].astype(int).values,
                      "개업÷점포_%": (hot_f / s0[hot_f.index] * 100).round(0).values,
                      "일괄폐업": bulk.reindex(hot_f.index).fillna(0).astype(int).values})
elig = s0[s0.rank(pct=True) >= C.TOPK_SEARCH_MIN_LEVEL_PCT].index
reg = [k for k in elig if mass.get(k, 0) > 0 and bulk.get(k, 0) > 0]          # 재등록으로 보이는 동
em = growth[[k for k in elig if k not in reg]].sort_values(ascending=False).head(N)
em_f = pd.DataFrame({"dong_code8": em.index, "동": [name(k) for k in em.index],
                     "점포증가율_%": (em * 100).round(1).values,
                     "점포_전": s0[em.index].astype(int).values, "점포_후": s1[em.index].astype(int).values,
                     "개업": o12[em.index].astype(int).values, "폐업": c12[em.index].astype(int).values,
                     "전년증가율_%": (growth_p[em.index] * 100).round(1).values,
                     "같은날10곳이상개업": mass.reindex(em.index).fillna(0).astype(int).values})
# 순증률(일괄 폐업 제외)로 보면 상위였지만 재등록으로 보여 뺀 동
nr_drop = (o12 - W("C_v_drop").loc[F_WIN[0]:F_WIN[1]].sum()) / s0
ex = nr_drop[[k for k in reg]].sort_values(ascending=False)
ex = ex[(ex >= 0.05) & (ex > growth[ex.index] + 0.05)]
ex_f = pd.DataFrame({"dong_code8": ex.index, "동": [name(k) for k in ex.index],
                     "순증률_v_drop_%": (ex * 100).round(1).values, "점포증가율_%": (growth[ex.index] * 100).round(1).values,
                     "같은날10곳이상개업": mass[ex.index].astype(int).values, "일괄폐업": bulk[ex.index].astype(int).values})

for fn, df in (("search_hot", hot_s), ("search_emerging", em_s), ("fnb_hot", hot_f), ("fnb_emerging", em_f),
               ("fnb_excluded_reregistration", ex_f)):
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(R / f"07_hotspots_{fn}.csv", encoding="utf-8-sig", index_label="순위")
    print(f"== {fn}\n{df.drop(columns='dong_code8').to_string()}\n")

# ---------------- 그림: 추이 ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ma = sh.rolling(3).mean() * 100
for k, c, ls in [(hot_s.dong_code8[1], "#c0504d", "-"), (hot_s.dong_code8[2], "#2a6f97", "-"),
                 (em_s.dong_code8[1], "#e0a33b", "-"), (em_s.dong_code8[2], "#e0a33b", "--"),
                 (em_s.dong_code8[5], "#6a994e", "-")]:
    x = ma[k].dropna()
    ax[0].plot(x.index, x.values, color=c, ls=ls, label=name(k))
ax[0].set_title("검색 비중 3개월 평균 (%) — 현재 핫스팟 2곳 · 떠오르는 곳 3곳", fontsize=10)
idx = S.loc["2022-01":F_END]
base = idx.loc[F_PREV]
seoul = idx.sum(axis=1) / base.sum() * 100
ax[1].plot(seoul.index, seoul.values, color="k", lw=2, label="서울 전체")
for k, c in zip(em_f.dong_code8[:4], ["#c0504d", "#2a6f97", "#e0a33b", "#6a994e"]):
    ax[1].plot(idx.index, idx[k] / base[k] * 100, color=c, label=name(k))
ax[1].axhline(100, color="grey", lw=.5)
ax[1].set_title(f"영업 점포 수 ({F_PREV} = 100) — 떠오르는 곳 4곳", fontsize=10)
for a in ax:
    t = [m for m in a.get_lines()[0].get_xdata() if str(m).endswith("-01")]
    a.set_xticks(t)
    a.set_xticklabels([str(m)[:4] for m in t])
    a.legend(fontsize=7)
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(R / "07_hotspot_trends.png", dpi=140)
log_run("07_hotspots", "핫스팟·떠오르는 곳 리스트 (기술 통계)", search_end=S_NOW, fnb_end=F_END)
