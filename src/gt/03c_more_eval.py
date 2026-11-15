"""추가 모델 평가: 자료 이상 처리 전후, 새 모델, 작년 같은 주와의 결합
- 정답: 원래 값(주 평가)과 보정 값(참고) 두 가지
- 결합: 작년 같은 주(보정)와 각 모델의 1:1 평균, 그리고 새 모델 전체 평균. 가중치 고정
출력: reports/gt_midterm/tables/t7_more_models.csv, t7_more_models_tests.csv, t7_more_models_clean_truth.csv
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
fac = pd.read_csv(C.RESULTS / "clean_factor.csv", index_col=0, parse_dates=True).factor
T["yc"] = T.y / T.week_end.map(fac).fillna(1.0)
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_*.parquet"))], ignore_index=True)
P["key"] = P.model
base = ["snaive", "naive", "arima", "sarima"]
new = ["snaive_c", "arima_c", "sarima_c", "arima_fourier", "stl_ets", "theta"]
P = P[(P.xset == "none") & P.model.isin(base + new) & P.unit.isin(com)].drop_duplicates(["origin", "unit", "week_end", "key"])
W = P.pivot_table(index=["origin", "unit", "week_end"], columns="key", values="y_hat").reset_index().merge(T, on=["unit", "week_end"]).dropna(subset=base + new)
LAB = {"snaive": "작년 같은 주", "naive": "Naive", "arima": "ARIMA", "sarima": "SARIMA",
       "snaive_c": "작년 같은 주 (보정)", "arima_c": "ARIMA (보정)", "sarima_c": "SARIMA (보정)", "arima_fourier": "ARIMA + 푸리에 (보정)",
       "stl_ets": "STL + ETS (보정)", "theta": "Theta (보정)"}
for k in ["arima", "arima_c", "arima_fourier", "stl_ets", "theta", "sarima_c"]:
    src = "snaive" if k == "arima" else "snaive_c"
    W[f"결합: {LAB[src]} + {LAB[k]}"] = (W[src] + W[k]) / 2
    LAB[f"결합: {LAB[src]} + {LAB[k]}"] = f"결합: {LAB[src]} + {LAB[k]}"
W["결합: 새 모델 5개 평균"] = W[["arima_c", "sarima_c", "arima_fourier", "stl_ets", "theta"]].mean(axis=1)
W["결합: 작년 같은 주(보정) + 새 모델 5개"] = W[["snaive_c", "arima_c", "sarima_c", "arima_fourier", "stl_ets", "theta"]].mean(axis=1)
cols = base + new + [c for c in W.columns if str(c).startswith("결합")]
W["o"] = W.origin.dt.date.astype(str)


def table(truth):
    R = W.groupby("o").apply(lambda g: pd.Series({c: (g[truth] - g[c]).abs().sum() / g[truth].sum() * 100 for c in cols}), include_groups=False)
    R.loc["4시점 평균"] = R.mean(); R.loc["2시점 평균(이상 구간 제외)"] = R.loc[["2025-01-26", "2025-04-27"]].mean()
    R = R.T; R.index = [LAB.get(i, i) for i in R.index]
    return R.round(2)


R = table("y"); R.to_csv(C.TABLES / "t7_more_models.csv", encoding="utf-8-sig")
table("yc").to_csv(C.TABLES / "t7_more_models_clean_truth.csv", encoding="utf-8-sig")
G = W.groupby(["o", "unit"]).apply(lambda d: pd.Series({c: (d.y - d[c]).abs().sum() / d.y.sum() for c in cols}), include_groups=False)
rows = []
for c in cols:
    if c == "snaive": continue
    for b in ("snaive", "arima"):
        if c == b: continue
        rows.append((LAB.get(c, c), LAB[b], (G[c] < G[b]).mean(), stats.wilcoxon(G[c], G[b]).pvalue))
pd.DataFrame(rows, columns=["model", "vs", "improved_share", "wilcoxon_p"]).round(4).to_csv(C.TABLES / "t7_more_models_tests.csv", index=False, encoding="utf-8-sig")
log_run("03c_more_eval", f"best4={R['4시점 평균'].idxmin()} {R['4시점 평균'].min():.2f}")
pd.set_option("display.width", 220)
print(R.sort_values("4시점 평균").to_string())
