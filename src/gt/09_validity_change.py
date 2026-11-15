"""G.T 타당성 검증 2: 변화가 같이 가나 (전체 동)
모든 값은 '서울 대비 비중'으로 바꿔 서울 전체 흐름과 동 크기를 뺀다.
A. 동 사이: 1년·2년 변화율(같은 달끼리)의 동별 순위 상관
   - 2024 vs 2023 (1~7월), 2025 vs 2024 (1~7월), 2025 vs 2023 (1~7월). 2024-08~10 이상 구간을 피해 1~7월만 쓴다
B. 동 안: 동마다 월별 '비중의 전년 같은 달 대비 변화' 시계열(2024-01~2025-07, 19개월)끼리 상관
   - 기준(귀무): 같은 계산을 다른 동의 자료와 짝지었을 때 (1000번 무작위 짝)
비교 자료: 지하철 하차(교통카드), 검색 비중(네이버), 음식점 영업 점포(인허가). 생활인구는 참고
출력: reports/gt_midterm/tables/v2_*.csv, fig/V2_*.png
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from common import log_run, MERGE, LP_RENAME, LP_SPLIT_PARENT, dong_list, set_korean_font

rng = np.random.default_rng(C.SEED)
to_unit = lambda c: MERGE.get(c, c)

# ---- 월별 동 값 ----
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
T["month"] = T.week_end.dt.to_period("M").astype(str)
gt = T.groupby(["unit", "month"]).y.mean().unstack()

sub = pd.read_parquet(C.RAW_SUB / "subway_daily.parquet")
sub["month"] = sub.date.dt.to_period("M").astype(str)
sm = sub.groupby(["station", "month"]).alight.mean().unstack()
lk = pd.read_csv(C.STA_LINK, encoding="utf-8-sig", dtype={"dong_code8": str})
lk["station"] = lk.station.replace({"서울역": "서울"})
lk = lk.groupby(["station", "dong_code8"], as_index=False).share_1000.max()
land = dong_list().set_index("dong_code8").land_km2
lk["w"] = lk.share_1000 * lk.dong_code8.map(land); lk = lk[lk.w > 0 & lk.station.isin(sm.index)]
lk = lk[lk.station.isin(sm.index)]
lk["w"] = lk.w / lk.groupby("station").w.transform("sum")
lk["unit"] = lk.dong_code8.map(to_unit)
subway = sum_ = None
parts = [sm.loc[r.station] * r.w for r in lk.itertuples()]
subway = pd.DataFrame(parts, index=lk.unit.values).groupby(level=0).sum(min_count=1)

S = pd.read_csv(C.SEARCH_CSV, encoding="utf-8-sig", dtype={"dong_code8": str})
S = S[S.tier == "전체"].assign(unit=lambda d: d.dong_code8.map(to_unit))
search = S.groupby(["unit", "month"]).seoul_share.sum(min_count=1).unstack()

F = pd.read_csv(C.FNB_CSV, encoding="utf-8-sig", dtype={"code": str})
F = F[F.level == "dong"].assign(unit=lambda d: d.code.map(to_unit))
fnb = F.groupby(["unit", "month"])["영업점포"].sum().unstack()

lp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RAW_LP / "lp_*.parquet")))], ignore_index=True)
lp["dong_code8"] = lp.dong_code8.replace(LP_RENAME)
for parent, kids in LP_SPLIT_PARENT.items():
    if parent in set(lp.dong_code8):
        par = lp[lp.dong_code8 == parent]
        lp = pd.concat([lp[lp.dong_code8 != parent]] + [par.assign(dong_code8=k, lp_2039=par.lp_2039 / len(kids)) for k in kids])
lp["unit"] = lp.dong_code8.map(to_unit); lp["month"] = lp.date.dt.to_period("M").astype(str)
lpop = lp.groupby(["unit", "date", "month"]).lp_2039.sum().groupby(["unit", "month"]).mean().unstack()

units = gt.index
def share(df):
    df = df.reindex(units)
    return df / df.sum(axis=0, min_count=1)
SR = {"생활이동": share(gt), "지하철 하차": share(subway), "검색 비중": share(search), "음식점 수": share(fnb), "생활인구 20~39세": share(lpop)}
OTH = ["지하철 하차", "검색 비중", "음식점 수", "생활인구 20~39세"]

# ---- A. 동 사이: 변화율 순위 상관 ----
def per(df, y, ms=range(1, 8)):
    cols = [f"{y}-{m:02d}" for m in ms]; cols = [c for c in cols if c in df.columns]
    return df[cols].mean(axis=1, skipna=False) if len(cols) == len(list(ms)) else pd.Series(np.nan, index=df.index)
rowsA, chg = [], {}
for a_, b_ in (("2023", "2024"), ("2024", "2025"), ("2023", "2025")):
    for k, df in SR.items():
        chg[(k, a_, b_)] = per(df, b_) / per(df, a_) - 1
    g = chg[("생활이동", a_, b_)]
    for o in OTH:
        d = pd.concat([g, chg[(o, a_, b_)]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 30: continue
        r, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        # 크게 는 동(생활이동 변화 상위 10%) 중 다른 자료도 서울 평균보다 는 비율, 하위 10%와 비교
        q_hi, q_lo = d.iloc[:, 0].quantile(.9), d.iloc[:, 0].quantile(.1)
        up_hi = (d[d.iloc[:, 0] >= q_hi].iloc[:, 1] > d.iloc[:, 1].median()).mean()
        up_lo = (d[d.iloc[:, 0] <= q_lo].iloc[:, 1] > d.iloc[:, 1].median()).mean()
        rowsA.append((f"{a_}→{b_}", o, len(d), r, p, up_hi, up_lo))
A = pd.DataFrame(rowsA, columns=["기간(1~7월)", "자료", "동 수", "변화 순위 상관", "p", "생활이동 많이 는 동 중 같이 는 비율", "생활이동 많이 준 동 중 는 비율"])
A.round(4).to_csv(C.TABLES / "v2_change_cross.csv", index=False, encoding="utf-8-sig")

# ---- B. 동 안: 전년 같은 달 대비 비중 변화의 월별 상관 ----
months = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13) if f"{y}-{m:02d}" <= "2025-07"]
def yoy(df):
    prev = [f"{int(m[:4]) - 1}{m[4:]}" for m in months]
    return np.log(df.reindex(columns=months).values / df.reindex(columns=prev).values)
Y = {k: yoy(v) for k, v in SR.items()}
def rowcorr(a, b):
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        m = np.isfinite(a[i]) & np.isfinite(b[i])
        if m.sum() >= 12 and np.std(a[i][m]) > 0 and np.std(b[i][m]) > 0: out[i] = np.corrcoef(a[i][m], b[i][m])[0, 1]
    return out
rowsB, dists = [], {}
for o in OTH:
    r = rowcorr(Y["생활이동"], Y[o]); ok = np.isfinite(r)
    null = []
    for _ in range(200):
        perm = rng.permutation(Y[o].shape[0])
        null.append(np.nanmedian(rowcorr(Y["생활이동"], Y[o][perm])))
    null = np.array(null)
    dists[o] = r[ok]
    rowsB.append((o, int(ok.sum()), np.nanmedian(r), (r[ok] > 0).mean(), np.mean(null), np.percentile(null, 97.5), (np.sum(null >= np.nanmedian(r)) + 1) / (len(null) + 1)))
B = pd.DataFrame(rowsB, columns=["자료", "동 수", "동별 상관 중앙값", "상관이 양수인 동 비율", "다른 동과 짝지은 기준 중앙값", "기준 97.5%", "순열 p"])
B.round(4).to_csv(C.TABLES / "v2_change_within.csv", index=False, encoding="utf-8-sig")

# ---- 그림 ----
set_korean_font()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5})
f, axs = plt.subplots(1, 4, figsize=(15, 3.9))
for ax, o in zip(axs, OTH):
    d = pd.concat([chg[("생활이동", "2023", "2025")], chg[(o, "2023", "2025")]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d.abs() < 3).all(axis=1)]
    ax.scatter(d.iloc[:, 1] * 100, d.iloc[:, 0] * 100, s=9, color="#6E86A8", alpha=.6, lw=0)
    ax.axhline(0, color="#8A8F98", lw=.6); ax.axvline(0, color="#8A8F98", lw=.6)
    r = A[(A["기간(1~7월)"] == "2023→2025") & (A["자료"] == o)]
    ax.set_title(f"{o}\n2년 변화 순위 상관 {r['변화 순위 상관'].iloc[0]:.2f}", loc="left")
    ax.set_xlabel(f"{o} 서울 대비 비중 변화 (%)"); qx = (d.iloc[:, 1] * 100).quantile([.01, .99]); ax.set_xlim(qx.iloc[0] - 3, qx.iloc[1] + 3); ax.set_ylim(-60, 110)
axs[0].set_ylabel("생활이동 서울 대비 비중 변화 (%)")
f.suptitle("G.T 타당성 2-A: 2년(2023→2025, 1~7월) 동안 생활이동 비중이 는 동은 다른 자료 비중도 늘었나 (전체 동, 점 하나 = 동 하나)", x=0.01, ha="left", fontsize=12)
f.tight_layout(); f.savefig(C.FIG / "V2_change_cross.png", bbox_inches="tight"); plt.close(f)

f, ax = plt.subplots(figsize=(8.5, 3.6))
bp = ax.boxplot([dists[o] for o in OTH], tick_labels=OTH, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#E3E9F3", color="#8A8F98"), medianprops=dict(color="#D4462A", lw=1.8))
for i, o in enumerate(OTH, 1):
    b = B.set_index("자료").loc[o]
    ax.hlines(b["다른 동과 짝지은 기준 중앙값"], i - .35, i + .35, color="#1C2430", ls="--", lw=1.4)
ax.plot([], [], color="#1C2430", ls="--", label="기준: 다른 동 자료와 짝지었을 때"); ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.axhline(0, color="#8A8F98", lw=.6)
ax.set_ylabel("동별 상관 (월별 전년 대비 변화)")
ax.set_title("G.T 타당성 2-B: 한 동 안에서 생활이동과 다른 자료가 같이 오르내리나 (2024-01~2025-07)", loc="left")
f.tight_layout(); f.savefig(C.FIG / "V2_change_within.png", bbox_inches="tight"); plt.close(f)

log_run("09_validity_change", "; ".join(f"{r['자료']} within={r['동별 상관 중앙값']:.2f}" for _, r in B.iterrows()))
print(A.round(3).to_string(index=False)); print(B.round(3).to_string(index=False))
