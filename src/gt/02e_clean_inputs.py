"""보정 자료에서 입력 다시 시험 (2026-11-16)
기본 모델 3개 (보정 자료, 02c와 같은 차수 선택 규칙), 같은 차수에 입력만 더한다
  - ax  : ARIMA(보정) + 입력            (= ARIMAX)
  - afx : ARIMA + 푸리에(보정) + 입력
  - sx  : SARIMA(보정, 52주 계절 차분) + 입력 (입력도 52주 차분, 02_models와 같은 방식) (= SARIMAX)
STL+ETS, Theta는 입력을 넣는 구조가 아니라 뺐다
입력 조합 7개: 검색, 지하철, 개폐업(순개업률), 둘씩 3개, 셋 다. 입력 규칙은 기존과 같다 (검색·개폐업 4개월 전 달, 지하철 13주 전, 학습 구간으로 표준화)
출력: results/gt/pred_cleanx_<origin>.parquet (model = ax / afx / sx, xset = 조합 이름)
"""
import sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
import importlib.util
spec = importlib.util.spec_from_file_location("m2c", str(C.ROOT / "src/gt/02c_more_models.py")); m2c = importlib.util.module_from_spec(spec)
sys_argv = list(sys.argv); sys.argv = [sys.argv[0]]; spec.loader.exec_module(m2c); sys.argv = sys_argv

XS = ["search", "subway", "fnb", "search_subway", "search_fnb", "subway_fnb", "search_subway_fnb"]
X = pd.read_parquet(C.INTERIM / "features_weekly.parquet")
D = X.merge(m2c.T[["unit", "week_end", "yc"]], on=["unit", "week_end"], how="left").sort_values(["unit", "week_end"])


def run_unit(u, g):
    g = g.set_index("week_end"); lyc = np.log1p(g.yc)
    zc = lyc - lyc.shift(C.SEASON)
    out = []
    for o in pd.to_datetime(C.ORIGINS):
        tr = lyc[lyc.index <= o].dropna()
        fut = g.index[g.index > o][:C.H]
        if len(fut) < C.H or len(tr) < 60: continue
        ztr = zc[zc.index <= o].dropna()
        lag = lyc.reindex(fut - pd.Timedelta(weeks=C.SEASON)).values
        Ftr, Ffu = m2c.fourier(tr.index), m2c.fourier(fut)
        bases = {}
        b = m2c.fit_best(tr.values);            bases["ax"] = b[1] if b else None
        b = m2c.fit_best(tr.values, Ftr);       bases["afx"] = b[1] if b else None
        b = m2c.fit_best(ztr.values) if len(ztr) >= 30 else None; bases["sx"] = b[1] if b else None
        for xs in XS:
            cols = C.X_SETS[xs]
            Xg = g[cols].copy()
            if Xg.isna().all().any(): continue
            Xg = Xg.bfill().ffill()
            mu, sd = Xg[Xg.index <= o].mean(), Xg[Xg.index <= o].std().replace(0, 1)
            Xs = (Xg - mu) / sd
            for m, order in bases.items():
                if order is None: continue
                try:
                    tr_ = "c" if order[1] == 0 else "n"
                    if m == "ax":
                        r = SARIMAX(tr.values, exog=Xs.reindex(tr.index).values, order=order, trend=tr_).fit(disp=False, maxiter=200)
                        pred = np.asarray(r.get_forecast(C.H, exog=Xs.reindex(fut).values).predicted_mean)
                    elif m == "afx":
                        r = SARIMAX(tr.values, exog=np.column_stack([Ftr, Xs.reindex(tr.index).values]), order=order, trend=tr_).fit(disp=False, maxiter=200)
                        pred = np.asarray(r.get_forecast(C.H, exog=np.column_stack([Ffu, Xs.reindex(fut).values])).predicted_mean)
                    else:
                        Xd = Xs - Xs.shift(C.SEASON)
                        r = SARIMAX(ztr.values, exog=Xd.reindex(ztr.index).fillna(0).values, order=order, trend=tr_).fit(disp=False, maxiter=200)
                        pred = lag + np.asarray(r.get_forecast(C.H, exog=Xd.reindex(fut).fillna(0).values).predicted_mean)
                except Exception:
                    continue
                out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "model": m, "xset": xs, "y_hat": np.expm1(pred)}))
    return out


if __name__ == "__main__":
    t0 = time.time()
    units = list(D.unit.unique())
    if len(sys.argv) > 1: units = units[: int(sys.argv[1])]
    res = Parallel(n_jobs=-1)(delayed(run_unit)(u, D[D.unit == u]) for u in units)
    P = pd.concat([d for r in res for d in r], ignore_index=True)
    for o, gg in P.groupby("origin"): gg.to_parquet(C.RESULTS / f"pred_cleanx_{pd.Timestamp(o).date()}.parquet", index=False)
    msg = f"units={len(units)}; rows={len(P)}; sec={time.time() - t0:.0f}"
    log_run("02e_clean_inputs", msg); print(msg)
