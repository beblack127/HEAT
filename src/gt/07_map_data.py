"""지도 '핫플지수' 화면 자료 -> data/processed/gt/hot_map.json
- 주별 생활이동(y)과 서울 대비 비중(s), 427동 -> 분석 단위 대응, 떠오르는 곳 규모 하한(2023년 고정)
- 대표 모델: 결합 = 작년 같은 주(보정) + ARIMA + 푸리에(보정), 1:1 (03c 평가에서 4시점·이상 없는 2시점 모두 상위)
- 비교 모델: 우리 자료(입력)를 넣은 ARIMAX 중 4시점 평균 WAPE가 가장 낮은 조합 (03b 평가)
- 실험 4개 시점 예측과 서울 전체 WAPE, 마지막 주 이후 13주 실시간 예측 (대표·비교 모델 모두)
"""
import glob, json, importlib.util, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import units_of_dongs, MERGE

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
t7 = pd.read_csv(C.TABLES / "t7_more_models.csv", index_col=0)
t6 = pd.read_csv(C.TABLES / "t6_input_combo.csv", index_col=0)
REP_ROW = "결합: 작년 같은 주 (보정) + ARIMA + 푸리에 (보정)"
label = "작년 같은 주 + ARIMA·푸리에 결합"
XL = {"search": "검색", "subway": "지하철", "fnb": "개폐업", "search_subway": "검색·지하철", "search_fnb": "검색·개폐업", "subway_fnb": "지하철·개폐업", "search_subway_fnb": "검색·지하철·개폐업"}
cand = [k for k in t6.index if k.startswith("arimax|") and k.split("|")[1] in XL]
cmp_key = min(cand, key=lambda k: t6.loc[k, "4시점 평균"])
cmp_x = cmp_key.split("|")[1]
label2 = f"ARIMAX + {XL[cmp_x]}"
col_o = [c for c in t7.columns if c[:2] == "20"]
wtab = {"main": {o: float(t7.loc[REP_ROW, o]) for o in col_o}, "cmp": {o: float(t6.loc[cmp_key, o]) for o in col_o},
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
exp, exp2 = exp_of("comb_fourier", "none"), exp_of("arimax", cmp_x)

# 실시간 예측
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, str(C.ROOT / "src/gt" / file)); m = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]; spec.loader.exec_module(m); return m
m2c, m02 = load("m2c", "02c_more_models.py"), load("m02", "02_models.py")
o_last = str(last.date())
R1 = Parallel(n_jobs=-1)(delayed(m2c.run_unit)(u, m2c.T[m2c.T.unit == u], [o_last]) for u in m2c.T.unit.unique())
F1 = pd.concat([d for o, _ in R1 for d in o], ignore_index=True)
F1 = F1.pivot_table(index=["unit", "week_end", "h"], columns="model", values="y_hat").reset_index()
F1["rep"] = (F1.snaive_c + F1.arima_fourier) / 2
R2 = Parallel(n_jobs=-1)(delayed(m02.run_unit)(u, m02.D[m02.D.unit == u], [o_last], [("arimax", cmp_x)]) for u in m02.D.unit.unique())
F2 = pd.concat([d for o, _ in R2 for d in o], ignore_index=True)
F2 = F2[(F2.model == "arimax") & (F2.xset == cmp_x)]
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
