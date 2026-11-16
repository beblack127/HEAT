"""전체 경우의 수 평가 (02f 결과)
- 모든 모델 x 결합(작년 같은 주와 1:1, 같은 자료) 안 함/함
- 13주 WAPE: 시점별, 4시점 평균, 이상 구간 없는 2시점 평균. 정답은 원래 값
- 검색 효과: 같은 설정에서 검색 없음 대비 (원래 지수 / 비중) WAPE 차이, 좋아진 동 비율, Wilcoxon
- 사다리(0~5단계) 표
출력: reports/gt_midterm/tables/t8_grid_all.csv, t8_grid_search_effect.csv, t8_ladder.csv
"""
import glob
import numpy as np, pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "grid/pred_*.parquet"))], ignore_index=True)
P = P[P.unit.isin(com)].merge(T, on=["unit", "week_end"])
P["o"] = P.origin.dt.date.astype(str)
sn = P[P.fam == "snaive"][["data", "o", "unit", "week_end", "y_hat"]].rename(columns={"y_hat": "sn"})
M = P[P.fam != "snaive"].merge(sn, on=["data", "o", "unit", "week_end"])
M["y_comb"] = (M.y_hat + M.sn) / 2
KEY = ["data", "fam", "order", "search", "lag13"]

# 동·시점 오차
def unit_err(df, col):
    g = df.assign(ae=(df.y - df[col]).abs()).groupby(KEY + ["o", "unit"])[["ae", "y"]].sum()
    return g
U0, U1 = unit_err(M, "y_hat"), unit_err(M, "y_comb")
def wape_tab(U):
    w = U.groupby(KEY + ["o"]).sum(); w = (w.ae / w.y * 100).unstack("o")
    w["4시점"] = w.mean(axis=1); w["2시점(이상 제외)"] = w[["2025-01-26", "2025-04-27"]].mean(axis=1)
    return w
W0, W1 = wape_tab(U0), wape_tab(U1)
W0["comb"], W1["comb"] = 0, 1
A = pd.concat([W0, W1]).reset_index().set_index(KEY + ["comb"]).round(2)
snw = sn.merge(T, on=["unit", "week_end"]).assign(ae=lambda d: (d.y - d.sn).abs()).groupby(["data", "o"])[["ae", "y"]].sum()
snw = (snw.ae / snw.y * 100).unstack("o"); snw["4시점"] = snw.mean(axis=1); snw["2시점(이상 제외)"] = snw[["2025-01-26", "2025-04-27"]].mean(axis=1)
A.to_csv(C.TABLES / "t8_grid_all.csv", encoding="utf-8-sig")
snw.round(2).to_csv(C.TABLES / "t8_snaive.csv", encoding="utf-8-sig")

# 검색 효과
rows = []
for comb, U in ((0, U0), (1, U1)):
    uw = (U.ae / U.y).unstack(["search"])
    for k, g in uw.groupby(level=["data", "fam", "order", "lag13"]):
        for s in ("idx", "share"):
            if s not in g.columns or "none" not in g.columns: continue
            d = g[[s, "none"]].dropna()
            if len(d) < 20: continue
            w = A.loc[(k[0], k[1], k[2], s, k[3], comb), "4시점"] - A.loc[(k[0], k[1], k[2], "none", k[3], comb), "4시점"]
            w2 = A.loc[(k[0], k[1], k[2], s, k[3], comb), "2시점(이상 제외)"] - A.loc[(k[0], k[1], k[2], "none", k[3], comb), "2시점(이상 제외)"]
            o427 = A.loc[(k[0], k[1], k[2], s, k[3], comb), "2025-04-27"] - A.loc[(k[0], k[1], k[2], "none", k[3], comb), "2025-04-27"]
            rows.append((*k, comb, s, round(w, 2), round(w2, 2), round(o427, 2), round((d[s] < d["none"]).mean(), 3), stats.wilcoxon(d[s], d["none"]).pvalue))
E = pd.DataFrame(rows, columns=["data", "fam", "order", "lag13", "comb", "search", "ΔWAPE 4시점", "ΔWAPE 2시점", "ΔWAPE 2025-04-27", "좋아진 동·시점", "wilcoxon_p"])
E.to_csv(C.TABLES / "t8_grid_search_effect.csv", index=False, encoding="utf-8-sig")

# 사다리
steps = [("0 팀원 설정", ("raw", "s13", "fixed", "idx", 1, 0)), ("1 계절 52주", ("raw", "s52", "fixed", "idx", 1, 0)), ("2 차수 AIC", ("raw", "s52", "aic", "idx", 1, 0)),
         ("3 검색 비중", ("raw", "s52", "aic", "share", 1, 0)), ("4 자료 보정", ("clean", "s52", "aic", "share", 1, 0)), ("5 작년 같은 주와 결합", ("clean", "s52", "aic", "share", 1, 1))]
L = []
for name, k in steps:
    base = (k[0], k[1], k[2], "none", k[4], k[5])
    L.append((name, *[A.loc[k, c] for c in ("2025-04-27", "4시점", "2시점(이상 제외)")], *[A.loc[base, c] for c in ("2025-04-27", "4시점", "2시점(이상 제외)")]))
LD = pd.DataFrame(L, columns=["단계", "검색O 04-27", "검색O 4시점", "검색O 2시점", "검색X 04-27", "검색X 4시점", "검색X 2시점"])
LD.to_csv(C.TABLES / "t8_ladder.csv", index=False, encoding="utf-8-sig")
log_run("03f_full_grid_eval", f"best4={A['4시점'].idxmin()} {A['4시점'].min():.2f}")
pd.set_option("display.width", 250)
print(LD.to_string(index=False)); print(A.sort_values("4시점").head(15).to_string()); print(snw.round(2))
