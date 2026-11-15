"""지도 '핫플지수' 탭 자료 -> data/processed/gt/hot_map.json
- 주별 생활이동(y)과 서울 대비 비중(s), 427동 -> 분석 단위 대응
- 실험 4개 시점의 대표 모델 예측
- 마지막 주 이후 13주 실시간 예측 (대표 모델). 평가가 아니라 서비스용 예측이다
대표 모델: 4개 시점 평균 WAPE가 가장 낮은 모델 중 작년 같은 주(snaive)를 뺀 것. snaive는 모든 동의 예측 증가율이 0이라 떠오르는 곳을 가릴 수 없다
"""
import glob, json, importlib.util, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import units_of_dongs, MERGE, unit_names

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
W = pd.read_parquet(C.RESULTS / "wape_by_origin.parquet")
rank = W.groupby("key").wape.mean().drop(labels=["snaive|none"], errors="ignore").sort_values()
best = rank.index[0]
m_best, x_best = best.split("|")
LAB = {"naive": "Naive", "arima": "ARIMA", "sarima": "SARIMA", "arimax": "ARIMAX", "sarimax": "SARIMAX"}
XL = {"none": "", "search": " + 검색", "all_noLP": " + 전체(생활인구 제외)", "all": " + 전체", "search_idx": " + 검색(원지수)"}
label = LAB[m_best] + XL.get(x_best.replace("+py", ""), "") + (" + 작년 같은 주" if "+py" in x_best else "")

weeks = sorted(T.week_end.unique()); last = pd.Timestamp(weeks[-1])
r3 = lambda v: None if v is None or not np.isfinite(v) else float(f"{v:.4g}")
yw = T.pivot(index="week_end", columns="unit", values="y").reindex(weeks)
sw = T.pivot(index="week_end", columns="unit", values="share").reindex(weeks)
unit = {u: {"y": [r3(v) for v in yw[u].values], "s": [r3(v) for v in sw[u].values]} for u in yw.columns}

# 실험 예측
P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RESULTS / "pred_*.parquet")))], ignore_index=True)
Pb = P[(P.model == m_best) & (P.xset == x_best)]
exp = {}
for (u, o), g in Pb.groupby(["unit", "origin"]):
    exp.setdefault(u, {})[str(pd.Timestamp(o).date())] = [r3(v) for v in g.sort_values("h").y_hat]

# 실시간 예측: 02_models.run_unit 재사용
spec = importlib.util.spec_from_file_location("m02", str(C.ROOT / "src/gt/02_models.py")); m02 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]; spec.loader.exec_module(m02)
variants = [] if x_best == "none" else [(m_best, x_best)]
res = Parallel(n_jobs=-1)(delayed(m02.run_unit)(u, m02.D[m02.D.unit == u], [str(last.date())], variants) for u in m02.D.unit.unique())
F = pd.concat([df for o, _ in res for df in o], ignore_index=True)
F = F[(F.model == (m_best if x_best != "none" else m_best)) & (F.xset == x_best)]
fc_weeks = [str(pd.Timestamp(w).date()) for w in sorted(F.week_end.dropna().unique())]
fc = {u: [r3(v) for v in g.sort_values("h").y_hat] for u, g in F.groupby("unit")}

d2u = units_of_dongs()
merged = {d: "신설동·용두동은 생활이동 자료에서 나뉘기 전(용신동) 코드로만 있어 둘을 합쳐 같은 값을 보여요." for d in MERGE}
out = {"weeks": [str(pd.Timestamp(w).date()) for w in weeks], "unit": unit, "d2u": d2u, "merged": merged,
       "fc": {"origin": str(last.date()), "model": label, "weeks": fc_weeks, "y": fc},
       "exp": {"origins": C.ORIGINS, "model": label, "y": exp}}
dst = C.ROOT / "data/processed/gt"; dst.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dst / "hot_map.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("대표 모델:", label, "| 실시간 예측", fc_weeks[0], "~", fc_weeks[-1], "| 동", len(fc), "| 크기", round((dst / "hot_map.json").stat().st_size / 1e6, 2), "MB")
