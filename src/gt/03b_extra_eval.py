"""추가 실험 평가: 입력 없음 / 검색 / 지하철 / 음식점 개폐업 / 검색+지하철 / 검색+지하철+음식점 x ARIMA·SARIMA 계열, 그리고 각 모델과 작년 같은 주의 1:1 결합.
출력: reports/gt_midterm/tables/t6_input_combo.csv, t6_input_combo_tests.csv
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_*.parquet"))]); P["key"] = P.model + "|" + P.xset
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
keys = ["snaive|none"] + [f"{m}|none" for m in ("arima", "sarima")] + [f"{m}x|{x}" for m in ("arima", "sarima") for x in ("search", "subway", "fnb", "search_subway", "search_fnb", "subway_fnb", "search_subway_fnb")]
P = P[P.key.isin(keys) & P.unit.isin(com)].drop_duplicates(["origin", "unit", "week_end", "key"])
W = P.pivot_table(index=["origin", "unit", "week_end"], columns="key", values="y_hat").reset_index().merge(T, on=["unit", "week_end"]).dropna(subset=keys)
for k in keys[1:]: W["결합|" + k] = (W["snaive|none"] + W[k]) / 2   # 작년 같은 주와 1:1, 가중치 고정
cols = keys + ["결합|" + k for k in keys[1:]]
W["o"] = W.origin.dt.date.astype(str)
R = W.groupby("o").apply(lambda g: pd.Series({c: (g.y - g[c]).abs().sum() / g.y.sum() * 100 for c in cols}), include_groups=False)
R.loc["4시점 평균"] = R.mean(); R.loc["2시점 평균(이상 구간 제외)"] = R.loc[["2025-01-26", "2025-04-27"]].mean()
R.T.round(2).to_csv(C.TABLES / "t6_input_combo.csv", encoding="utf-8-sig")
G = W.groupby(["o", "unit"]).apply(lambda d: pd.Series({c: (d.y - d[c]).abs().sum() / d.y.sum() for c in cols}), include_groups=False)
rows = []
for m in ("arima", "sarima"):
    for x in ("search", "subway", "fnb", "search_subway", "search_fnb", "subway_fnb", "search_subway_fnb"):
        for pre in ("", "결합|"):
            a, b = f"{pre}{m}x|{x}", f"{pre}{m}|none"
            rows.append((a, b, (G[a] < G[b]).mean(), stats.wilcoxon(G[a], G[b]).pvalue))
for k in [c for c in cols if c.startswith("결합|")]:
    rows.append((k, "snaive|none", (G[k] < G["snaive|none"]).mean(), stats.wilcoxon(G[k], G["snaive|none"]).pvalue))
pd.DataFrame(rows, columns=["model", "vs", "improved_share", "wilcoxon_p"]).round(4).to_csv(C.TABLES / "t6_input_combo_tests.csv", index=False, encoding="utf-8-sig")
log_run("03b_extra_eval", f"best4={R.loc['4시점 평균'].idxmin()} {R.loc['4시점 평균'].min():.2f}")
print(R.T.round(2).to_string())
