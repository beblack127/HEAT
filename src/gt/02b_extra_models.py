"""추가 실험: 검색 + 지하철, 지하철만 (생활인구 제외). ARIMAX·SARIMAX, 같은 4개 시점·같은 차수 규칙.
02_models.run_unit을 그대로 쓰고 결과는 results/gt/pred_extra_<시점>.parquet 로 따로 저장한다.
"""
import importlib.util, sys, time
import pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

ARGV = list(sys.argv)
spec = importlib.util.spec_from_file_location("m02", str(C.ROOT / "src/gt/02_models.py")); m02 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]; spec.loader.exec_module(m02)
XS = ARGV[1].split(",") if len(ARGV) > 1 else ["subway", "search_subway"]
TAG = ARGV[2] if len(ARGV) > 2 else "extra"
VAR = [(m, xs) for m in ("arimax", "sarimax") for xs in XS]
t0 = time.time()
res = Parallel(n_jobs=-1)(delayed(m02.run_unit)(u, m02.D[m02.D.unit == u], None, VAR) for u in m02.D.unit.unique())
P = pd.concat([df for o, _ in res for df in o], ignore_index=True)
P = P[P.xset.isin(XS)]
for o, g in P.groupby("origin"):
    g.to_parquet(C.RESULTS / f"pred_{TAG}_{pd.Timestamp(o).date()}.parquet", index=False)
msg = f"rows={len(P)}; fallback={int(P.fallback.sum())}; sec={time.time() - t0:.0f}"
log_run("02b_extra_models", f"{XS}; " + msg); print(msg)
