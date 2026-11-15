"""G.T 타당성 검증 1: 동별 생활이동 수준이 다른 독립 자료와 같은 곳을 가리키나
기간: 2025-01 ~ 2025-08 (품질 점검 구간 2024-08~10, 2025-09 이후를 피함)
비교 자료 (동별 기간 평균, 같은 427동 단위):
  - 지하철 하차 (OA-12914, 역 -> 동 가중 배분)      : 독립 자료 (교통카드)
  - 검색 서울 대비 비중 (네이버 데이터랩)            : 독립 자료 (관심)
  - 음식점 영업 점포 수 (인허가)                   : 독립 자료 (공급)
  - 생활인구 20~39세 11~21시 (OA-14991)           : 같은 통신 자료 계열이라 참고용
지표: 스피어만 순위 상관, 현재 핫스팟(상위 10%) 겹침 비율 (무작위면 10%)
출력: reports/gt_midterm/tables/v1_*.csv, fig/V1_*.png
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from common import log_run, MERGE, LP_RENAME, LP_SPLIT_PARENT, dong_list, unit_names, set_korean_font

A, B = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31")
to_unit = lambda c: MERGE.get(c, c)
NM = unit_names()

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
t = T[(T.week_end >= A) & (T.week_end <= B)]
gt = t.groupby("unit").y.mean().rename("생활이동")

# 지하철 하차: 역 하루 평균 하차 x 동 가중
sub = pd.read_parquet(C.RAW_SUB / "subway_daily.parquet")
sub = sub[(sub.date >= A) & (sub.date <= B)].groupby("station").alight.mean()
lk = pd.read_csv(C.STA_LINK, encoding="utf-8-sig", dtype={"dong_code8": str})
lk["station"] = lk.station.replace({"서울역": "서울"})
lk = lk.groupby(["station", "dong_code8"], as_index=False).share_1000.max()
land = dong_list().set_index("dong_code8").land_km2
lk["w"] = lk.share_1000 * lk.dong_code8.map(land); lk = lk[lk.w > 0]
lk["w"] = lk.w / lk.groupby("station").w.transform("sum")
lk["v"] = lk.station.map(sub) * lk.w
subway = lk.dropna(subset=["v"]).assign(unit=lambda d: d.dong_code8.map(to_unit)).groupby("unit").v.sum().rename("지하철 하차")

# 검색 비중
S = pd.read_csv(C.SEARCH_CSV, encoding="utf-8-sig", dtype={"dong_code8": str})
S = S[(S.tier == "전체") & (S.month >= "2025-01") & (S.month <= "2025-08")].assign(unit=lambda d: d.dong_code8.map(to_unit))
search = S.groupby(["unit", "month"]).seoul_share.sum(min_count=1).groupby("unit").mean().rename("검색 비중")

# 음식점 영업 점포
F = pd.read_csv(C.FNB_CSV, encoding="utf-8-sig", dtype={"code": str})
F = F[(F.level == "dong") & (F.month >= "2025-01") & (F.month <= "2025-08")].assign(unit=lambda d: d.code.map(to_unit))
fnb = F.groupby(["unit", "month"])["영업점포"].sum().groupby("unit").mean().rename("음식점 수")

# 생활인구 20~39세
lp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RAW_LP / "lp_2025*.parquet")))], ignore_index=True)
lp = lp[(lp.date >= A) & (lp.date <= B)]
lp["dong_code8"] = lp.dong_code8.replace(LP_RENAME)
for parent, kids in LP_SPLIT_PARENT.items():
    if parent in set(lp.dong_code8):
        par = lp[lp.dong_code8 == parent]
        lp = pd.concat([lp[lp.dong_code8 != parent]] + [par.assign(dong_code8=k, lp_2039=par.lp_2039 / len(kids)) for k in kids])
lpop = lp.assign(unit=lp.dong_code8.map(to_unit)).groupby(["unit", "date"]).lp_2039.sum().groupby("unit").mean().rename("생활인구 20~39세")

X = pd.concat([gt, subway, search, fnb, lpop], axis=1)
X = X[X.index.isin(gt.index)]
others = ["지하철 하차", "검색 비중", "음식점 수", "생활인구 20~39세"]
indep = {"지하철 하차": "교통카드", "검색 비중": "네이버 검색", "음식점 수": "인허가", "생활인구 20~39세": "같은 통신 자료 (참고)"}

rows = []
hot_gt = set(gt.sort_values(ascending=False).index[: int(round(len(gt) * C.HOT_TOP))])
for c in others:
    d = X[["생활이동", c]].dropna()
    d = d[d[c] > 0]
    r, p = stats.spearmanr(d["생활이동"], d[c])
    n_top = int(round(len(d) * C.HOT_TOP))
    top_c = set(d[c].sort_values(ascending=False).index[:n_top])
    top_g = set(d["생활이동"].sort_values(ascending=False).index[:n_top])
    rows.append((c, indep[c], len(d), r, p, len(top_g & top_c) / n_top, n_top / len(d)))
R = pd.DataFrame(rows, columns=["자료", "출처", "동 수", "스피어만 상관", "p", "핫스팟 겹침", "무작위 겹침"])
R.round(4).to_csv(C.TABLES / "v1_correlation.csv", index=False, encoding="utf-8-sig")

# 다른 자료끼리의 상관 (생활이동이 '가운데'에 있는지 보기)
M = X[["생활이동"] + others].rank().corr()
M.round(3).to_csv(C.TABLES / "v1_corr_matrix.csv", encoding="utf-8-sig")

# 핫스팟 43곳 중 독립 자료 3개 중 몇 개에서도 상위 10%인가
top_sets = {c: set(X[c].dropna().sort_values(ascending=False).index[: int(round(X[c].notna().sum() * C.HOT_TOP))]) for c in ["지하철 하차", "검색 비중", "음식점 수"]}
cnt = {u: sum(u in s for s in top_sets.values()) for u in hot_gt}
H = pd.DataFrame([(u, *NM.get(u, ("", u)), gt[u], cnt[u], *[int(u in top_sets[c]) for c in top_sets]) for u in hot_gt],
                 columns=["unit", "gu", "dong", "생활이동", "독립자료_상위10%_개수"] + [f"{c}_상위10%" for c in top_sets]).sort_values("생활이동", ascending=False)
H.round(1).to_csv(C.TABLES / "v1_hotspot_agreement.csv", index=False, encoding="utf-8-sig")
# 무작위 43곳이면 기대 개수
rnd = sum(len(s) / X[c].notna().sum() for c, s in top_sets.items())
dist = H["독립자료_상위10%_개수"].value_counts().reindex(range(4), fill_value=0)

# 그림
set_korean_font()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5})
f, axs = plt.subplots(1, 4, figsize=(15, 3.9))
for ax, c in zip(axs, others):
    d = X[["생활이동", c]].dropna(); d = d[d[c] > 0]
    hot = d.index.isin(hot_gt)
    ax.scatter(d[c][~hot], d["생활이동"][~hot], s=9, color="#9AA6B8", alpha=.7, lw=0, label="그 밖의 동")
    ax.scatter(d[c][hot], d["생활이동"][hot], s=16, color="#D4462A", alpha=.9, lw=0, label="현재 핫스팟 (생활이동 상위 10%)")
    ax.set_xscale("log"); ax.set_yscale("log")
    r = R.set_index("자료").loc[c]
    ax.set_title(f"{c}  ·  순위 상관 {r['스피어만 상관']:.2f}\n핫스팟 겹침 {r['핫스팟 겹침'] * 100:.0f}% (무작위 {r['무작위 겹침'] * 100:.0f}%)", loc="left")
    ax.set_xlabel(c + (" (같은 통신 자료)" if c.startswith("생활인구") else ""))
axs[0].set_ylabel("생활이동 20~39세 목적 4+5+7 (일평균)")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
f.suptitle("G.T 타당성 1: 생활이동이 많은 동은 다른 자료에서도 많다 (2025-01~08 평균, 동 하나 = 점 하나, 로그 축)", x=0.01, ha="left", fontsize=12)
f.tight_layout(); f.savefig(C.FIG / "V1_validity_scatter.png", bbox_inches="tight"); plt.close(f)

f, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(range(4), dist.values, color=["#C9CED6", "#E8A07F", "#D4462A", "#7A2513"])
for k, v in enumerate(dist.values): ax.text(k, v + .4, str(v), ha="center", fontsize=10)
ax.set_xticks(range(4), ["0개", "1개", "2개", "3개 모두"])
ax.set_xlabel("지하철 하차·검색 비중·음식점 수 중 상위 10%에도 든 자료 수")
ax.set_ylabel("현재 핫스팟 동 수")
ax.set_title(f"현재 핫스팟 {len(H)}곳: 평균 {H['독립자료_상위10%_개수'].mean():.1f}개 자료에서도 상위 10% (무작위면 {rnd:.1f}개)", loc="left")
f.tight_layout(); f.savefig(C.FIG / "V1_hotspot_agreement.png", bbox_inches="tight"); plt.close(f)

log_run("08_validity", "; ".join(f"{a}={b:.2f}" for a, b in zip(R["자료"], R["스피어만 상관"])))
print(R.round(3).to_string(index=False))
print("핫스팟 독립 자료 일치 분포:", dist.to_dict(), "평균", round(H["독립자료_상위10%_개수"].mean(), 2), "무작위", round(rnd, 2))
print(M.round(2).to_string())
print(H.head(15)[["gu", "dong", "독립자료_상위10%_개수"]].to_string(index=False))
print(H[H["독립자료_상위10%_개수"] == 0][["gu", "dong"]].to_string(index=False))
