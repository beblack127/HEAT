"""가짜 입력 실험 평가 (02g 결과)
팀원 설정(고정 차수, 보정 전, 13주 전 생활이동 입력 포함)에서 입력마다 13주 WAPE
- 시점별, 4시점 평균, 2025-04-27
- 입력 없음 대비 차이, 좋아진 동·시점 비율, Wilcoxon
출력: reports/gt_midterm/tables/t10_placebo.csv, t10_placebo_tests.csv
"""
import glob
import pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
com = set(pd.read_parquet(C.RESULTS / "unit_metrics.parquet").unit)
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "grid/placebo_*.parquet"))], ignore_index=True)
P = P[P.unit.isin(com)].merge(T, on=["unit", "week_end"])
P["o"] = P.origin.dt.date.astype(str)
# 모든 입력이 다 있는 동·시점만 (공정 비교)
k = P.groupby(["fam", "o", "unit"]).input.nunique()
full = k[k == P.input.nunique()].index
P = P.set_index(["fam", "o", "unit"]).loc[full].reset_index()
U = P.assign(ae=(P.y - P.y_hat).abs()).groupby(["fam", "input", "o", "unit"])[["ae", "y"]].sum()
W = U.groupby(["fam", "input", "o"]).sum(); W = (W.ae / W.y * 100).unstack("o")
W["4시점"] = W.mean(axis=1)
W.round(2).to_csv(C.TABLES / "t10_placebo.csv", encoding="utf-8-sig")
uw = (U.ae / U.y).unstack("input")
rows = []
for fam, g in uw.groupby(level="fam"):
    for inp in ("search_index", "placebo", "subway_exit_4w", "fnb_net3"):
        d = g[[inp, "none"]].dropna()
        d427 = d.xs("2025-04-27", level="o")
        rows.append((fam, inp, round(W.loc[(fam, inp), "2025-04-27"] - W.loc[(fam, "none"), "2025-04-27"], 2),
                     round(W.loc[(fam, inp), "4시점"] - W.loc[(fam, "none"), "4시점"], 2),
                     round((d427[inp] < d427["none"]).mean(), 3), stats.wilcoxon(d427[inp], d427["none"]).pvalue,
                     round((d[inp] < d["none"]).mean(), 3), stats.wilcoxon(d[inp], d["none"]).pvalue))
E = pd.DataFrame(rows, columns=["fam", "input", "ΔWAPE 04-27", "ΔWAPE 4시점", "좋아진 동 04-27", "p 04-27", "좋아진 동·시점 4시점", "p 4시점"])
E.to_csv(C.TABLES / "t10_placebo_tests.csv", index=False, encoding="utf-8-sig")
log_run("03g_placebo_eval", f"s13 placebo d0427={E[(E.fam=='s13')&(E.input=='placebo')]['ΔWAPE 04-27'].iloc[0]}")
pd.set_option("display.width", 250)
print(W.round(2).to_string()); print(E.to_string(index=False))
