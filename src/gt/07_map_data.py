"""지도 '핫플지수' 화면 자료 -> data/processed/gt/hot_map.json
- 주별 생활이동(y)과 서울 대비 비중(s), 427동 -> 분석 단위 대응, 떠오르는 곳 규모 하한(2023년 고정)
- 대표 모델: 결합 = 작년 같은 주(보정) + ARIMA + 푸리에(보정), 1:1 (03c 평가에서 4시점·이상 없는 2시점 모두 상위)
- 비교 모델: 대표 모델 + 검색량(서울 대비 비중, 4개월 전 달) = 작년 같은 주(보정) + ARIMA·푸리에·검색(보정), 1:1 (03e 평가, 2026-11-16 변경)
- 실험 4개 시점 예측과 서울 전체 WAPE, 마지막 주 이후 13주 실시간 예측 (대표·비교 모델 모두)
"""
import glob, json, importlib.util, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import units_of_dongs, MERGE

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
t7 = pd.read_csv(C.TABLES / "t7_more_models.csv", index_col=0)
t9 = pd.read_csv(C.TABLES / "t9_clean_inputs.csv", index_col=[0, 1, 2])
REP_ROW = "결합: 작년 같은 주 (보정) + ARIMA + 푸리에 (보정)"
label = "작년 같은 주 + ARIMA·푸리에 결합"
label2 = "작년 같은 주 + ARIMA·푸리에·검색량 결합"
CMP_ROW = (1, "ARIMA+푸리에 (보정)", "검색")
col_o = [c for c in t7.columns if c[:2] == "20"]
wtab = {"main": {o: float(t7.loc[REP_ROW, o]) for o in col_o}, "cmp": {o: float(t9.loc[CMP_ROW, o]) for o in col_o},
        "snaive": {o: float(t7.loc["작년 같은 주", o]) for o in col_o}, "arima": {o: float(t7.loc["ARIMA", o]) for o in col_o}}

weeks = sorted(T.week_end.unique()); last = pd.Timestamp(weeks[-1])
r3 = lambda v: None if v is None or not np.isfinite(v) else float(f"{v:.4g}")
yw = T.pivot(index="week_end", columns="unit", values="y").reindex(weeks)
sw = T.pivot(index="week_end", columns="unit", values="share").reindex(weeks)
unit = {u: {"y": [r3(v) for v in yw[u].values], "s": [r3(v) for v in sw[u].values]} for u in yw.columns}

P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_*.parquet"))], ignore_index=True)
def exp_of(m, x):
    d = {}
    for (u, o), g in P[(P.model == m) & (P.xset == x)].groupby(["unit", "origin"]):
        d.setdefault(u, {})[str(pd.Timestamp(o).date())] = [r3(v) for v in g.sort_values("h").y_hat]
    return d
exp = exp_of("comb_fourier", "none")
Q = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_cleanx_*.parquet"))])
Q = Q[(Q.model == "afx") & (Q.xset == "search")].merge(P[P.model == "snaive_c"][["origin", "unit", "week_end", "y_hat"]].rename(columns={"y_hat": "sn"}), on=["origin", "unit", "week_end"])
Q = Q.assign(y_hat=(Q.y_hat + Q.sn) / 2, model="cmp", xset="none")
P = pd.concat([P, Q[P.columns.intersection(Q.columns)]], ignore_index=True)
exp2 = exp_of("cmp", "none")

# 실시간 예측
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, str(C.ROOT / "src/gt" / file)); m = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]; spec.loader.exec_module(m); return m
m2c = load("m2c", "02c_more_models.py")
o_last = str(last.date())
R1 = Parallel(n_jobs=-1)(delayed(m2c.run_unit)(u, m2c.T[m2c.T.unit == u], [o_last]) for u in m2c.T.unit.unique())
F1 = pd.concat([d for o, _ in R1 for d in o], ignore_index=True)
F1 = F1.pivot_table(index=["unit", "week_end", "h"], columns="model", values="y_hat").reset_index()
F1["rep"] = (F1.snaive_c + F1.arima_fourier) / 2
XF = pd.read_parquet(C.INTERIM / "features_weekly.parquet")[["unit", "week_end"] + C.X_SETS["search"]]
def live_search(u, g):
    """대표 모델과 같은 차수(입력 없는 ARIMA+푸리에에서 AIC로 고름)에 검색 입력만 더함 (02e와 같은 규칙)"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    g = g.set_index("week_end"); lyc = np.log1p(g.yc); o = pd.Timestamp(o_last)
    tr = lyc[lyc.index <= o].dropna()
    fut = pd.DatetimeIndex([o + pd.Timedelta(weeks=k) for k in range(1, C.H + 1)])
    if len(tr) < 60: return None
    b = m2c.fit_best(tr.values, m2c.fourier(tr.index))
    if not b: return None
    x = XF[XF.unit == u].set_index("week_end")[C.X_SETS["search"]].reindex(g.index.union(fut)).bfill().ffill()
    if x.isna().all().any(): return None
    mu, sd = x[x.index <= o].mean(), x[x.index <= o].std().replace(0, 1); xs = (x - mu) / sd
    try:
        r = SARIMAX(tr.values, exog=np.column_stack([m2c.fourier(tr.index), xs.reindex(tr.index).values]), order=b[1], trend="c" if b[1][1] == 0 else "n").fit(disp=False, maxiter=200)
        pr = np.asarray(r.get_forecast(C.H, exog=np.column_stack([m2c.fourier(fut), xs.reindex(fut).values])).predicted_mean)
    except Exception:
        return None
    return pd.DataFrame({"unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "afs": np.expm1(pr)})
R2 = Parallel(n_jobs=-1)(delayed(live_search)(u, m2c.T[m2c.T.unit == u]) for u in m2c.T.unit.unique())
F2 = pd.concat([d for d in R2 if d is not None], ignore_index=True).merge(F1[["unit", "h", "snaive_c"]], on=["unit", "h"])
F2["y_hat"] = (F2.afs + F2.snaive_c) / 2
fc_weeks = [str(pd.Timestamp(w).date()) for w in sorted(F1.week_end.dropna().unique())]
fc = {u: [r3(v) for v in g.sort_values("h").rep] for u, g in F1.groupby("unit")}
fc2 = {u: [r3(v) for v in g.sort_values("h").y_hat] for u, g in F2.groupby("unit")}

base = T[T.week_end.dt.year == C.EMERGING_MIN_BASE_YEAR].groupby("unit").y.mean()
minlvl = float(base.quantile(C.EMERGING_MIN_SIZE_PCT))
d2u = units_of_dongs()
merged = {d: "신설동·용두동은 생활이동 자료에서 나뉘기 전(용신동) 코드로만 있어 둘을 합쳐 같은 값을 보여요." for d in MERGE}
out = {"minlvl": round(minlvl, 1), "minlvl_year": C.EMERGING_MIN_BASE_YEAR, "weeks": [str(pd.Timestamp(w).date()) for w in weeks], "unit": unit, "d2u": d2u, "merged": merged,
       "anomaly": {"from": "2024-08-01", "to": "2024-10-31", "factor": round(float(pd.read_csv(C.RESULTS / "clean_factor.csv", index_col=0).factor.max()), 3)},
       "fc": {"origin": o_last, "model": label, "weeks": fc_weeks, "y": fc, "model2": label2, "y2": fc2},
       "exp": {"origins": C.ORIGINS, "model": label, "y": exp, "model2": label2, "y2": exp2, "wape": wtab}}
dst = C.ROOT / "data/processed/gt"; dst.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dst / "hot_map.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("대표:", label, "| 비교:", label2, "| 실시간", fc_weeks[0], "~", fc_weeks[-1], "| 동", len(fc), len(fc2), "| MB", round((dst / "hot_map.json").stat().st_size / 1e6, 2))
