"""SARIMAX (statsmodels). 적용 범위: 검색 모든 동, 개폐업 서울·구·개폐업 많은 동.

차수: fold 학습 구간(TRAIN_START ~ 학습 끝)에서 AutoARIMA(계절 12)가 고른 (p,d,q)(P,D,Q). fold 안에서 고정.
외생변수 (미래 값이 필요한 것은 시차 h, 코로나 더미는 시차 없음)
  검색 동       covid, log1p(그 동 개업 3개월 평균)_{t-h}
  개폐업 동     covid, log(그 동 검색 비중 3개월 평균)_{t-h}  (검색 없는 동은 covid만)
  개폐업 구     covid, log(구 안 동 검색 비중 합 3개월 평균)_{t-h}
  개폐업 서울   covid, log(서울 검색 지수 합 3개월 평균)_{t-h}
  v_keep 폐업   + 일괄 폐업 있음(0/1)_{t-h}
h = 3, 6만 돌린다. h = 12는 검색 시차 12를 넣으면 F2024 학습 표본이 12개 미만이라 적합할 수 없다.
origin t마다 적합 구간(외생변수가 있는 첫 달 ~ t)으로 다시 적합해 h개월을 예측한다.
출력: results/forecast/pred_sarimax.parquet
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from common import C, Timer, covid, log_run, origins, train_end
import data as D

warnings.filterwarnings("ignore")
FINAL = "--final" in sys.argv
FOLDS = ["FINAL"] if FINAL else list(C.FOLDS)
H_SARIMAX = (3, 6)


def auto_orders(y, end, scale):
    """fold 학습 구간에서 AutoARIMA 차수."""
    x = (y.loc[C.TRAIN_START:end] * scale).reset_index().melt(id_vars=y.index.name or "month",
                                                               var_name="unique_id", value_name="y")
    x.columns = ["ds", "unique_id", "y"]
    x["ds"] = pd.PeriodIndex(x.ds, freq="M").to_timestamp()
    x = x.dropna()
    sf = StatsForecast(models=[AutoARIMA(season_length=C.SEASON)], freq="MS", n_jobs=C.N_JOBS)
    sf.fit(df=x)
    orders = {}
    for i, uid in enumerate(sf.uids):
        try:
            p, q, P, Q, s, d, Dd = sf.fitted_[i, 0].model_["arma"]
            orders[str(uid)] = ((p, d, q), (P, Dd, Q, s if (P or Dd or Q) else 0))
        except Exception:
            orders[str(uid)] = ((1, 0, 0), (0, 0, 0, 0))
    return orders


def fit_one(sid, y, X, order, sorder, origins_h, h, scale, how):
    import warnings as _w
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _w.filterwarnings("ignore")
    out = []
    for t in origins_h:
        yy = y.loc[:t].dropna() * scale
        XX = X.loc[yy.index]
        ok = XX.notna().all(1)
        yy, XX = yy[ok], XX[ok]
        if len(yy) < 12:
            continue
        XX = XX.loc[:, XX.std() > 0]          # 적합 구간에서 상수인 열(예: 코로나 이후 구간의 covid) 제외
        # 미래 h개월의 외생변수: 시차 h라 t까지의 값, covid는 정해진 값
        idx = [str(pd.Period(t, "M") + k) for k in range(1, h + 1)]
        Xf = X.reindex(idx)[XX.columns]
        if Xf.isna().any().any():
            continue
        so = sorder
        if so[3] and len(yy) < so[3] + 2 * (so[0] + so[1] + so[2]) + 6:
            so = (0, 0, 0, 0)
        pred = None
        for o, s_ in ((order, so), ((1, 0, 0), (0, 0, 0, 0))):
            try:
                ex = XX.values if XX.shape[1] else None
                m = SARIMAX(yy.values, exog=ex, order=o, seasonal_order=s_, trend="c" if o[1] == 0 else "n",
                            enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False, maxiter=200)
                pred = r.forecast(h, exog=Xf.values if ex is not None else None)
                # 발산 점검: 예측이 과거 범위(0 ~ 최댓값의 3배)를 벗어나면 다음 차수로
                hi = 3 * np.nanmax(yy.values) + 1e-9
                if np.all(np.isfinite(pred)) and pred.min() > -hi and pred.max() < hi:
                    break
                pred = None
            except Exception:
                pred = None
        if pred is None or not np.all(np.isfinite(pred)):
            continue
        v = pred.sum() if how == "sum" else pred.mean()
        out.append((sid, t, max(v / scale, 0.0)))
    return out


def exog_frames(target, level, h, final):
    """계열별 외생변수 표 (행 = 월)."""
    s, f = D.monthly(final)
    months = sorted(f.month.unique())
    cv = pd.Series(covid(months), index=months, dtype=float)
    res = {}
    if target == "search":
        O = D.wide("open", "dong", final).reindex(months)
        x = np.log1p(O.rolling(3).mean()).shift(h)
        for sid in D.wide("search", "dong", final).columns:
            res[sid] = pd.DataFrame({"covid": cv, "x": x[sid] if sid in x else np.nan})
        return res
    sh = D.search_wide("share", final=final).reindex(months)
    if level == "dong":
        sx = np.log(sh.rolling(3).mean()).shift(h)
    elif level == "gu":
        gu = D.gu_of(final)
        g = sh.T.groupby(gu.reindex(sh.columns).values).sum().T
        sx = np.log(g.rolling(3).mean()).shift(h)
    else:
        tot = D.search_wide("index", final=final).reindex(months).sum(1, min_count=1)
        sx = pd.DataFrame({"서울": np.log(tot.rolling(3).mean()).shift(h)})
    bulk = (D.fnb_wide("bulk_n", level, final).reindex(months) > 0).astype(float).shift(h)
    y = D.wide(target, level, final)
    for sid in y.columns:
        cols = {"covid": cv}
        if sid in sx.columns and sx[sid].notna().sum() > 0:
            cols["search"] = sx[sid]
        if target == "close_v_keep":
            cols["bulk"] = bulk[sid]
        res[sid] = pd.DataFrame(cols)
    return res


def main():
    rows = []
    for fold in FOLDS:
        end = train_end(fold)
        sc = json.load(open(C.INTERIM / f"scope_{fold}.json"))
        for target, levels in D.LEVELS.items():
            how = D.AGG[target]
            scale = C.SEARCH_SCALE if target == "search" else 1.0
            for level in levels:
                y = D.wide(target, level, FINAL)
                ids = list(y.columns) if (target == "search" or level != "dong") else \
                    [i for i in y.columns if i in sc["busy"]]
                with Timer(f"{fold} {target} {level} 차수 선택 ({len(ids)}계열)"):
                    orders = auto_orders(y[ids], end, scale)
                for h in H_SARIMAX:
                    og = origins(fold, h)
                    ex = exog_frames(target, level, h, FINAL)
                    with Timer(f"{fold} {target} {level} h={h}"):
                        res = Parallel(n_jobs=C.N_JOBS)(
                            delayed(fit_one)(sid, y[sid].loc[C.TRAIN_START:], ex[sid], *orders[str(sid)], og, h,
                                             scale, how) for sid in ids)
                    for r in res:
                        for sid, t, v in r:
                            rows.append((target, level, fold, h, sid, t, v))
    out = pd.DataFrame(rows, columns=["target", "level", "fold", "h", "series_id", "origin", "y_pred"])
    out["model"] = "SARIMAX"
    fn = "pred_sarimax_final.parquet" if FINAL else "pred_sarimax.parquet"
    out.to_parquet(C.RESULTS / fn, index=False)
    print(out.groupby(["target", "level", "fold", "h"]).size())
    log_run("04_sarimax", "SARIMAX (AutoARIMA 차수 + 외생 시차 h)", rows=len(out), final=FINAL)


if __name__ == "__main__":
    main()
