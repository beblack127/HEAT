"""대표 모델 vs 대표 모델 + 검색: 동별 오차 변화 (4시점 합친 13주 WAPE)
출력: reports/gt_midterm/tables/t11_search_by_dong.csv, t11_search_pattern.csv
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import config as C

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
B = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_more_*.parquet"))])
X = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_cleanx_*.parquet"))])
k = ["origin", "unit", "week_end"]
sn = B[B.model == "snaive_c"].set_index(k).y_hat; af = B[B.model == "arima_fourier"].set_index(k).y_hat
afs = X[(X.model == "afx") & (X.xset == "search")].set_index(k).y_hat
D = pd.DataFrame({"rep": (sn + af) / 2, "rep_s": (sn + afs) / 2}).dropna().reset_index()
D = D[D.unit.isin(com)].merge(T, on=["unit", "week_end"])
g = D.assign(a=(D.y - D.rep).abs(), b=(D.y - D.rep_s).abs()).groupby("unit")[["a", "b", "y"]].sum()
W = pd.DataFrame({"rep": g.a / g.y * 100, "rep_s": g.b / g.y * 100}); W["d"] = W.rep_s - W.rep
# 동 특성 (2024년까지, 예측 시점 전 정보)
F = pd.read_parquet(C.INTERIM / "features_weekly.parquet")
F = F[F.week_end <= "2024-10-27"].groupby("unit")[["search_index", "search_share"]].mean()
L = T[(T.week_end <= "2024-10-27") & (T.week_end > "2023-10-27")].groupby("unit").y.mean().rename("level")
W = W.join(F).join(L)
W.round(3).to_csv(C.TABLES / "t11_search_by_dong.csv", encoding="utf-8-sig")
rows = []
for c, nm in (("level", "생활이동 규모"), ("search_index", "검색 지수 크기")):
    q = pd.qcut(W[c].rank(method="first"), 3, labels=["하", "중", "상"])
    for lab, gg in W.groupby(q, observed=True):
        rows.append((nm, lab, len(gg), round((gg.d < 0).mean() * 100, 1), round(gg.d.median(), 2)))
    rho, p = stats.spearmanr(W[c], W.d, nan_policy="omit"); rows.append((nm, "순위상관", len(W), round(rho, 3), round(p, 4)))
P = pd.DataFrame(rows, columns=["특성", "구간", "동 수", "좋아진 동 %", "중앙값 Δ(또는 p)"])
P.to_csv(C.TABLES / "t11_search_pattern.csv", index=False, encoding="utf-8-sig")
print(len(W), "improved", round((W.d < 0).mean() * 100, 1), "median d", round(W.d.median(), 2), "mean rep", W.rep.mean().round(2))
print(W.d.describe().round(2)); print(P.to_string(index=False))
