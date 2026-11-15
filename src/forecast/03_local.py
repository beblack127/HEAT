"""국소 모델: AutoETS, AutoTheta, AutoARIMA(SARIMA, 계절 12), Croston/TSB.

origin t마다 TRAIN_START ~ t 자료로 계열마다 다시 적합하고 12개월을 예측한 뒤, 앞 h개월을 합(개수) 또는 평균(검색)한다.
적용 범위 (fold 학습 구간으로 정함)
  검색 동: ETS, Theta, ARIMA
  개폐업 서울·구: ETS, Theta, ARIMA
  개폐업 동: ETS, Theta 전부 / ARIMA 개폐업 많은 동 / Croston·TSB 0이 많은 동
출력: results/forecast/pred_local.parquet, data/interim/forecast/scope_<fold>.json
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import TSB, AutoARIMA, AutoETS, AutoTheta, CrostonClassic, Naive

from common import C, Timer, log_run, origins, train_end
import data as D

warnings.filterwarnings("ignore")
FINAL = "--final" in sys.argv
FOLDS = ["FINAL"] if FINAL else list(C.FOLDS)
ONLY = [a.split("=")[1] for a in sys.argv if a.startswith("--target=")]
C.RESULTS.mkdir(parents=True, exist_ok=True)


def scope(fold):
    """fold 학습 구간 기준 적용 범위."""
    end = train_end(fold)
    o = D.wide("open", "dong", FINAL).loc[C.TRAIN_START:end]
    c = D.wide("close_v_drop", "dong", FINAL).loc[C.TRAIN_START:end]
    busy = sorted(((o + c).mean() >= C.BUSY_DONG_MIN).pipe(lambda s: s[s].index))
    zero = ((o == 0).mean() >= C.CROSTON_ZERO_SHARE) | ((c == 0).mean() >= C.CROSTON_ZERO_SHARE)
    sc = {"busy": busy, "croston": sorted(zero[zero].index)}
    json.dump(sc, open(C.INTERIM / f"scope_{fold}.json", "w"))
    return sc


def to_long(y, end):
    x = y.loc[C.TRAIN_START:end]
    df = x.reset_index().melt(id_vars=x.index.name or "month", var_name="unique_id", value_name="y")
    df.columns = ["ds", "unique_id", "y"]
    df["ds"] = pd.PeriodIndex(df.ds, freq="M").to_timestamp()
    return df.dropna(subset=["y"])[["unique_id", "ds", "y"]]


def run(y, models, ids, origin, scale=1.0):
    if not ids:
        return None
    df = to_long(y[ids] * scale, origin)
    sf = StatsForecast(models=models, freq="MS", n_jobs=C.N_JOBS, fallback_model=Naive())
    fc = sf.forecast(df=df, h=C.MAX_H)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    fc["step"] = fc.groupby("unique_id").cumcount() + 1
    return fc


def collect(fc, target, level, fold, origin, how, scale):
    out = []
    mcols = [c for c in fc.columns if c not in ("unique_id", "ds", "step")]
    for h in C.HORIZONS:
        if origin not in origins(fold, h):
            continue
        g = fc[fc.step <= h].groupby("unique_id")[mcols]
        a = (g.sum() if how == "sum" else g.mean()) / scale
        a = a.clip(lower=0)
        q = a.reset_index().melt(id_vars="unique_id", var_name="model", value_name="y_pred")
        q = q.rename(columns={"unique_id": "series_id"})
        out.append(q.assign(target=target, level=level, fold=fold, origin=origin, h=h))
    return out


def M_full():
    return [AutoETS(season_length=C.SEASON, alias="ETS"), AutoTheta(season_length=C.SEASON, alias="Theta"),
            AutoARIMA(season_length=C.SEASON, alias="SARIMA")]


def main():
    rows = []
    for fold in FOLDS:
        sc = scope(fold)
        print(f"{fold}: 개폐업 많은 동 {len(sc['busy'])}, 0 많은 동 {len(sc['croston'])}", flush=True)
        for target, levels in D.LEVELS.items():
            if ONLY and target not in ONLY:
                continue
            how = D.AGG[target]
            scale = C.SEARCH_SCALE if target == "search" else 1.0
            for level in levels:
                y = D.wide(target, level, FINAL)
                ids = list(y.columns)
                with Timer(f"{fold} {target} {level} ({len(ids)}계열)"):
                    for origin in origins(fold, min(C.HORIZONS)):
                        if target == "search" or level != "dong":
                            fcs = [run(y, M_full(), ids, origin, scale)]
                        else:
                            fcs = [run(y, [AutoETS(season_length=C.SEASON, alias="ETS"),
                                           AutoTheta(season_length=C.SEASON, alias="Theta")], ids, origin),
                                   run(y, [AutoARIMA(season_length=C.SEASON, alias="SARIMA")],
                                       [i for i in ids if i in sc["busy"]], origin),
                                   run(y, [CrostonClassic(alias="Croston"), TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB")],
                                       [i for i in ids if i in sc["croston"]], origin)]
                        for fc in fcs:
                            if fc is not None:
                                rows += collect(fc, target, level, fold, origin, how, scale)
                        print(f"   origin {origin} 끝", flush=True)

    out = pd.concat(rows, ignore_index=True)
    tag = "_".join(ONLY) if ONLY else "all"
    fn = f"pred_local_{tag}{'_final' if FINAL else ''}.parquet"
    out.to_parquet(C.RESULTS / fn, index=False)
    print(out.groupby(["target", "level", "fold", "h", "model"]).size().unstack("model"))
    log_run("03_local", "ETS, Theta, SARIMA, Croston, TSB", rows=len(out), final=FINAL, targets=tag)


if __name__ == "__main__":
    main()
