"""기준선: Naive (최근 h개월 값 유지), Seasonal Naive (1년 전 같은 h개월).

검색은 h개월 평균, 개수는 h개월 합.
  naive_t   = agg(y_{t-h+1..t})
  snaive_t  = agg(y_{t+1-12..t+h-12})   (h = 12이면 naive와 같다)
출력: results/forecast/pred_baselines.parquet
"""
import sys

import pandas as pd

from common import C, log_run, origins
import data as D

FINAL = "--final" in sys.argv
FOLDS = ["FINAL"] if FINAL else list(C.FOLDS)
C.RESULTS.mkdir(parents=True, exist_ok=True)

rows = []
for target, levels in D.LEVELS.items():
    how = D.AGG[target]
    for level in levels:
        y = D.wide(target, level, final=FINAL)
        for h in C.HORIZONS:
            r = y.rolling(h, min_periods=h)
            naive = r.sum() if how == "sum" else r.mean()
            snaive = (r.sum() if how == "sum" else r.mean()).shift(12 - h)   # 끝이 t+h-12인 h개월
            for fold in FOLDS:
                og = [o for o in origins(fold, h) if o in y.index]
                for name, p in (("Naive", naive), ("SeasonalNaive", snaive)):
                    q = p.loc[og].stack().rename("y_pred").reset_index()
                    q.columns = ["origin", "series_id", "y_pred"]
                    q = q.assign(target=target, level=level, fold=fold, h=h, model=name)
                    rows.append(q)

out = pd.concat(rows, ignore_index=True)
fn = "pred_baselines_final.parquet" if FINAL else "pred_baselines.parquet"
out.to_parquet(C.RESULTS / fn, index=False)
print(out.groupby(["target", "level", "fold", "h", "model"]).size().unstack("model"))
log_run("02_baselines", "Naive, SeasonalNaive", rows=len(out), final=FINAL)
