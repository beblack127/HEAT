"""보정 자료 입력 실험 평가 (02e): ARIMA·ARIMA+푸리에·SARIMA (보정) x 입력 없음 + 7조합, 결합(작년 같은 주(보정)와 1:1)
출력: reports/gt_midterm/tables/t9_clean_inputs.csv, t9_clean_inputs_tests.csv
"""
import glob
import pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
X = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_cleanx_*.parquet"))])
B = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_more_*.parquet"))])
base = {"arima_c": "ax", "arima_fourier": "afx", "sarima_c": "sx"}
B0 = B[B.model.isin(base)].assign(model=lambda d: d.model.map(base), xset="none")
SN = B[B.model == "snaive_c"][["origin", "unit", "week_end", "y_hat"]].rename(columns={"y_hat": "sn"})
P = pd.concat([X, B0[X.columns]]).merge(SN, on=["origin", "unit", "week_end"]).merge(T, on=["unit", "week_end"])
P = P[P.unit.isin(com)]
P["o"] = P.origin.dt.date.astype(str); P["yc"] = (P.y_hat + P.sn) / 2
LAB = {"ax": "ARIMA", "afx": "ARIMA+푸리에", "sx": "SARIMA"}
XL = {"none": "입력 없음", "search": "검색", "subway": "지하철", "fnb": "개폐업", "search_subway": "검색+지하철", "search_fnb": "검색+개폐업", "subway_fnb": "지하철+개폐업", "search_subway_fnb": "셋 다"}
rows, U = [], {}
for comb, col in ((0, "y_hat"), (1, "yc")):
    g = P.assign(ae=(P.y - P[col]).abs()).groupby(["model", "xset", "o", "unit"])[["ae", "y"]].sum()
    U[comb] = (g.ae / g.y).unstack("xset")
    w = g.groupby(["model", "xset", "o"]).sum(); w = (w.ae / w.y * 100).unstack("o")
    w["4시점"] = w.mean(axis=1); w["2시점(이상 제외)"] = w[["2025-01-26", "2025-04-27"]].mean(axis=1); w["결합"] = comb
    rows.append(w)
R = pd.concat(rows).reset_index()
R["모델"] = R.model.map(LAB) + " (보정)"; R["입력"] = R.xset.map(XL)
R = R.set_index(["결합", "모델", "입력"]).drop(columns=["model", "xset"]).round(2)
R.to_csv(C.TABLES / "t9_clean_inputs.csv", encoding="utf-8-sig")
tt = []
for comb in (0, 1):
    u = U[comb]
    for m in base.values():
        um = u.xs(m, level="model")
        for x in XL:
            if x == "none" or x not in um.columns: continue
            d = um[[x, "none"]].dropna()
            tt.append((comb, LAB[m], XL[x], round((d[x] < d["none"]).mean(), 3), stats.wilcoxon(d[x], d["none"]).pvalue))
pd.DataFrame(tt, columns=["결합", "모델", "입력", "좋아진 동·시점", "wilcoxon_p"]).round(4).to_csv(C.TABLES / "t9_clean_inputs_tests.csv", index=False, encoding="utf-8-sig")
log_run("03e_clean_inputs_eval", f"best={R['4시점'].idxmin()} {R['4시점'].min():.2f}")
pd.set_option("display.width", 220)
print(R[["4시점", "2시점(이상 제외)", "2025-04-27"]].unstack("입력")["4시점"].round(2).to_string())
print(R[["2시점(이상 제외)"]].unstack("입력")["2시점(이상 제외)"].round(2).to_string())
