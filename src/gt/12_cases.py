"""발표용 사례 4곳 고르기 (2025-04-27 시점, 13주 예측)
- 대표 모델(작년 같은 주(보정) + ARIMA·푸리에(보정) 1:1 결합): 동별 WAPE 최저·최고
- 대표 모델 + 검색: 대표 모델 대비 WAPE가 가장 많이 준 동·가장 많이 는 동
대상: 공통 416동 중 규모 하한(2023년 하위 25%) 이상인 동 (예측 시점 직전 13주 평균)
출력: reports/gt_midterm/tables/case4.csv, results/gt/case4_series.parquet
"""
import glob
import numpy as np, pandas as pd
import config as C

O = pd.Timestamp("2025-04-27")
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
floor = pd.read_csv(C.TABLES / "emerging_min_level.csv", index_col=0).loc["min_level", "value"]
lvl = T[(T.week_end <= O) & (T.week_end > O - pd.Timedelta(weeks=13))].groupby("unit").y.mean()
units = [u for u in com if lvl.get(u, 0) >= floor]

B = pd.read_parquet(C.RESULTS / f"pred_more_{O.date()}.parquet")
X = pd.read_parquet(C.RESULTS / f"pred_cleanx_{O.date()}.parquet")
sn = B[B.model == "snaive_c"].set_index(["unit", "week_end"]).y_hat
af = B[B.model == "arima_fourier"].set_index(["unit", "week_end"]).y_hat
afs = X[(X.model == "afx") & (X.xset == "search")].set_index(["unit", "week_end"]).y_hat
D = pd.DataFrame({"sn": sn, "rep": (sn + af) / 2, "rep_s": (sn + afs) / 2}).dropna().reset_index()
D = D[D.unit.isin(units)].merge(T, on=["unit", "week_end"])
w = D.assign(a=(D.y - D.rep).abs(), b=(D.y - D.rep_s).abs(), c=(D.y - D.sn).abs()).groupby("unit")[["a", "b", "c", "y"]].sum()
W = pd.DataFrame({"rep": w.a / w.y * 100, "rep_s": w.b / w.y * 100, "sn": w.c / w.y * 100})
W["d"] = W.rep_s - W.rep

names = pd.read_csv(C.ROOT / "data/processed/nametags/01_행정동_목록.csv", encoding="utf-8-sig", dtype=str)
print(names.columns.tolist()[:8])
pick = [("rep_best", W.rep.idxmin()), ("rep_worst", W.rep.idxmax()), ("search_best", W.d.idxmin()), ("search_worst", W.d.idxmax())]
out = pd.DataFrame([(r, u, *W.loc[u, ["rep", "rep_s", "sn", "d"]].round(2), round(lvl[u])) for r, u in pick],
                   columns=["role", "unit", "wape_rep", "wape_rep_search", "wape_lastyear", "d_search", "level_13w"])
out["n_units"] = len(W)
out["rank_rep"] = [int(W.rep.rank().loc[u]) for u in out.unit]
out.to_csv(C.TABLES / "case4.csv", index=False, encoding="utf-8-sig")
D[D.unit.isin(out.unit)].to_parquet(C.RESULTS / "case4_series.parquet", index=False)
print(out.to_string(index=False))
print("search better share:", round((W.d < 0).mean(), 3), "| median d:", round(W.d.median(), 2))

# 판정: 예측(대표 모델·대표+검색)과 실제로 본 순위. 떠오르는 곳은 규모 하한 이상 동 중 순위
yw = T.pivot(index="week_end", columns="unit", values="y")
Tsh = pd.read_parquet(C.INTERIM / "target_weekly.parquet").pivot(index="week_end", columns="unit", values="share")
w = yw.index[(yw.index > O) & (yw.index <= O + pd.Timedelta(weeks=C.H))]
prev = w - pd.Timedelta(weeks=C.SEASON)
base_sh = Tsh.reindex(prev).mean()
F = pd.DataFrame({"sn": sn, "rep": (sn + af) / 2, "rep_s": (sn + afs) / 2}).reset_index()
F = F[F.unit.isin(com)]
J = []
for col in ("rep", "rep_s", "actual"):
    g = yw.reindex(w)[list(com)] if col == "actual" else F.pivot(index="week_end", columns="unit", values=col).reindex(w)
    lv = g.mean(); gr = g.div(g.sum(axis=1), axis=0).mean() / base_sh.reindex(g.columns) - 1
    grk = gr[gr.index.isin(units)].rank(ascending=False)
    for u in out.unit:
        J.append((u, col, int(lv.rank(ascending=False)[u]), int(grk.get(u, np.nan)) if u in grk else None, round(gr[u] * 100, 1), len(lv), len(grk)))
J = pd.DataFrame(J, columns=["unit", "by", "hot_rank", "emerging_rank", "growth_pct", "n_hot", "n_em"])
J.to_csv(C.TABLES / "case4_judge.csv", index=False, encoding="utf-8-sig")
print(J.to_string(index=False))
