"""평가 (명세 §6): WAPE, MASE, 동별 개선 비율, Wilcoxon, DM-HLN. 표는 reports/gt_midterm/tables/."""
import glob
import numpy as np, pandas as pd
from scipy import stats
import config as C
from common import log_run

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RESULTS / "pred_*.parquet")))], ignore_index=True)
P = P.merge(T, on=["unit", "week_end"], how="left").dropna(subset=["y", "y_hat"])
P["ae"] = (P.y - P.y_hat).abs()
P["key"] = P.model + "|" + P.xset

# 동마다 학습 구간 작년 같은 주 MAE (MASE 분모)
T = T.sort_values(["unit", "week_end"])
T["ae52"] = (T.y - T.groupby("unit").y.shift(C.SEASON)).abs()
scale = {o: T[T.week_end <= o].groupby("unit").ae52.mean() for o in pd.to_datetime(C.ORIGINS)}

# 공통 동 집합: 시점마다 모든 (모델, 묶음) 예측이 있는 동
keys_main = [f"{m}|{x}" for m in ("naive", "snaive", "arima", "sarima") for x in ("none",)] + \
            [f"{m}|{x}" for m in ("arimax", "sarimax") for x in ("search", "all_noLP", "all")]
common = {}
for o, g in P.groupby("origin"):
    have = g.groupby("unit").key.agg(set)
    common[o] = set(have[have.apply(lambda s: set(keys_main) <= s)].index)
P["in_common"] = [u in common.get(o, set()) for u, o in zip(P.unit, P.origin)]
C_ = P[P.in_common]

def wape(g): return g.ae.sum() / g.y.sum() * 100

# 동별 13주 WAPE와 MASE
U = C_.groupby(["origin", "key", "unit"]).apply(lambda g: pd.Series({"wape": wape(g), "mae": g.ae.mean()}), include_groups=False).reset_index()
U["mase"] = [m / scale[o].get(u, np.nan) for m, o, u in zip(U.mae, U.origin, U.unit)]

# 표 1: 주 결과표 (시점 평균 WAPE), 모델 x 묶음
W = C_.groupby(["origin", "key"]).apply(wape, include_groups=False).rename("wape").reset_index()
W[["model", "xset"]] = W.key.str.split("|", expand=True)
avg = W.groupby(["model", "xset"]).wape.mean().unstack()
order_m = ["naive", "snaive", "arima", "sarima", "arimax", "sarimax"]
order_x = ["none", "search", "all_noLP", "all"]
tab1 = pd.DataFrame(index=order_m, columns=order_x, dtype=float)
for m in order_m:
    for x in order_x:
        mm = {"arimax": "arima", "sarimax": "sarima"}.get(m, m) if x == "none" else m
        if x != "none" and m in ("naive", "snaive", "arima", "sarima"): continue
        if (mm, x) in [(a, b) for a, b in zip(avg.stack().index.get_level_values(0), avg.stack().index.get_level_values(1))]:
            tab1.loc[m, x] = avg.loc[mm, x]
tab1.round(2).to_csv(C.TABLES / "t1_main_wape.csv", encoding="utf-8-sig")

# 표 2: 시점별 WAPE
tab2 = W.pivot(index="origin", columns="key", values="wape")
tab2.index = [pd.Timestamp(o).date() for o in tab2.index]
tab2.round(2).to_csv(C.TABLES / "t2_by_origin_wape.csv", encoding="utf-8-sig")

# MASE 요약 (동별 중앙값, 시점 평균)
mase = U.groupby(["key"]).mase.median().rename("mase_median").to_frame()
mase["mase_mean"] = U.groupby("key").mase.mean()
mase.round(3).to_csv(C.TABLES / "t_mase.csv", encoding="utf-8-sig")

# 개선 비율과 Wilcoxon: 각 키 vs snaive, 외생변수 있음 vs 없음
def paired(a, b):
    m = U[U.key == a].merge(U[U.key == b], on=["origin", "unit"], suffixes=("_a", "_b"))
    if len(m) < 10: return np.nan, np.nan, len(m)
    d = m.wape_a - m.wape_b
    p = stats.wilcoxon(m.wape_a, m.wape_b).pvalue if (d != 0).any() else 1.0
    return (d < 0).mean(), p, len(m)

rows = []
for k in sorted(U.key.unique()):
    if k == "snaive|none": continue
    r, p, n = paired(k, "snaive|none"); rows.append((k, "snaive|none", r, p, n))
for m, base in (("arimax", "arima|none"), ("sarimax", "sarima|none")):
    for x in ("search", "all_noLP", "all"):
        r, p, n = paired(f"{m}|{x}", base); rows.append((f"{m}|{x}", base, r, p, n))
for x in ("none", "search", "all_noLP", "all"):   # M3
    r, p, n = paired(f"arimax|{x}+py", "arima|none" if x == "none" else f"arimax|{x}"); rows.append((f"arimax|{x}+py", "arima|none" if x == "none" else f"arimax|{x}", r, p, n))
for m in ("arimax", "sarimax"):                   # M4
    r, p, n = paired(f"{m}|search", f"{m}|search_idx"); rows.append((f"{m}|search", f"{m}|search_idx", r, p, n))
tests = pd.DataFrame(rows, columns=["model", "vs", "improved_share", "wilcoxon_p", "n_pairs"])
tests.round(4).to_csv(C.TABLES / "t_tests.csv", index=False, encoding="utf-8-sig")

# DM-HLN: 서울 전체 주별 절대오차 합 (시점 x 13주)
def dm_hln(e1, e2, h=C.H):
    d = np.asarray(e1) - np.asarray(e2); n = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    v = (gam[0] + 2 * sum(gam[1:])) / n
    if v <= 0: return np.nan, np.nan
    s = dbar / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * stats.t.sf(abs(s), df=n - 1)
agg = C_.groupby(["key", "origin", "h"]).ae.sum().unstack("key")
dm = []
for k in agg.columns:
    if k == "snaive|none": continue
    s, p = dm_hln(agg[k].values, agg["snaive|none"].values); dm.append((k, s, p, len(agg)))
pd.DataFrame(dm, columns=["model", "dm_stat_vs_snaive", "p", "n"]).round(4).to_csv(C.TABLES / "t_dm.csv", index=False, encoding="utf-8-sig")

# 표 3 (M3), 표 4 (M4)
t3 = []
for x in ("none", "search", "all_noLP", "all"):
    a = "arima|none" if x == "none" else f"arimax|{x}"; b = f"arimax|{x}+py"
    t3.append((x, W[W.key == a].wape.mean(), W[W.key == b].wape.mean(), W[W.key == "snaive|none"].wape.mean()))
pd.DataFrame(t3, columns=["xset", "without_prior_year", "with_prior_year", "snaive"]).round(2).to_csv(C.TABLES / "t3_prior_year.csv", index=False, encoding="utf-8-sig")
t4 = [(m, W[W.key == f"{m}|search_idx"].wape.mean(), W[W.key == f"{m}|search"].wape.mean()) for m in ("arimax", "sarimax")]
pd.DataFrame(t4, columns=["model", "search_raw_index", "search_seoul_share"]).round(2).to_csv(C.TABLES / "t4_search_share.csv", index=False, encoding="utf-8-sig")

# 표 5: 동별 WAPE 분포
q = U.groupby("key").wape.quantile([.1, .25, .5, .75, .9]).unstack().round(2)
q.to_csv(C.TABLES / "t5_dong_wape_dist.csv", encoding="utf-8-sig")
U.to_parquet(C.RESULTS / "unit_metrics.parquet", index=False)
W.to_parquet(C.RESULTS / "wape_by_origin.parquet", index=False)
pd.Series({str(pd.Timestamp(o).date()): len(s) for o, s in common.items()}).to_csv(C.TABLES / "common_set_size.csv", header=["n_units"], encoding="utf-8-sig")
log_run("03_evaluate", f"common={ {str(pd.Timestamp(o).date()): len(s) for o, s in common.items()} }")
print(tab1.round(2)); print(tests.round(3).head(12))
