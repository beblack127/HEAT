"""모델 적합·예측 (명세 §5). 모든 모델은 log1p(y)를 적합한다.

- naive: 시점 직전 주 값
- snaive: 52주(364일) 전 값 (작년 같은 주)
- arima: (p,d,q) 격자에서 AIC 최소. p∈{0,1,2}, d∈{0,1}, q∈{0,1}
- sarima: SARIMA(p,d,q)(0,1,0)52. 계절 차분한 z_t = log y_t - log y_{t-52}에 ARMA를 적합하고,
          예측은 log y_t = log y_{t-52} + z_t 로 되돌린다 (계절 차분 = 작년 같은 주 + 보정)
- arimax / sarimax: 위 차수 그대로 + 외생변수 (학습 구간으로 표준화). sarimax의 외생변수도 52주 차분
- 차수는 외생변수 없는 모델에서 한 번 고르고, 외생변수 버전에 그대로 쓴다
- 외생변수 앞쪽 결측(자료 시작 전)은 그 동의 첫 관측값으로 채운다. 한 열이 통째로 없으면 그 묶음은 건너뛴다
출력: results/gt/pred_<origin>.parquet, results/gt/fit_log.csv
"""
import sys, warnings, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
X = pd.read_parquet(C.INTERIM / "features_weekly.parquet")
D = X.merge(T, on=["unit", "week_end"], how="left").sort_values(["unit", "week_end"])   # 입력 표에는 마지막 주 뒤 13주도 있다
XCOLS = sorted({c for v in C.X_SETS.values() for c in v} | {"search_index", C.PRIOR_YEAR_X})
GRID = [(p, d, q) for p in range(3) for d in range(2) for q in range(2)]

# 적합할 (모델, 묶음) 목록. +py = 작년 같은 주 입력 추가 (M3), search_idx = 원래 검색 지수 (M4)
VARIANTS = [(m, xs) for m in ("arimax", "sarimax") for xs in ("search", "all_noLP", "all")]
VARIANTS += [("arimax", f"{xs}+py") for xs in ("none", "search", "all_noLP", "all")]
VARIANTS += [(m, "search_idx") for m in ("arimax", "sarimax")]


def xcols_of(xs):
    if xs == "search_idx": return ["search_index"]
    base, py = (xs[:-3], True) if xs.endswith("+py") else (xs, False)
    return C.X_SETS[base] + ([C.PRIOR_YEAR_X] if py else [])


def fit_best(z, trend_ok=True):
    best = None
    for (p, d, q) in GRID:
        try:
            r = SARIMAX(z, order=(p, d, q), trend="c" if (d == 0 and trend_ok) else "n").fit(disp=False, maxiter=200)
            if np.isfinite(r.aic) and (best is None or r.aic < best[0]): best = (r.aic, (p, d, q))
        except Exception:
            pass
    return best[1] if best else (0, 1, 0)


def fit_pred(z, order, ex_tr=None, ex_fu=None, h=C.H):
    p, d, q = order
    r = SARIMAX(z, exog=ex_tr, order=order, trend="c" if d == 0 else "n").fit(disp=False, maxiter=200)
    return np.asarray(r.get_forecast(h, exog=ex_fu).predicted_mean), int(r.mle_retvals.get("converged", True)) if hasattr(r, "mle_retvals") and r.mle_retvals else 1


def run_unit(u, g, origins=None, variants=None):
    out, logs = [], []
    g = g.set_index("week_end")
    ly = np.log1p(g.y)
    for o in pd.to_datetime(origins or C.ORIGINS):
        tr = ly[ly.index <= o].dropna()
        fut = g.index[(g.index > o)][:C.H]
        if len(fut) < C.H or len(tr) < 60: continue
        rows = {}
        rows[("naive", "none")] = np.repeat(tr.iloc[-1], C.H)
        lag = ly.reindex(fut - pd.Timedelta(weeks=C.SEASON)).values
        rows[("snaive", "none")] = lag
        # 계절 차분 계열
        zs = (ly - ly.shift(C.SEASON))
        ztr = zs[zs.index <= o].dropna()
        try:
            o_a = fit_best(tr.values); rows[("arima", "none")], cv = fit_pred(tr.values, o_a)
        except Exception as e:
            o_a = (0, 1, 0); rows[("arima", "none")] = rows[("naive", "none")]; logs.append((u, o, "arima", "none", f"fallback {e}"))
        try:
            o_s = fit_best(ztr.values); zf, cv = fit_pred(ztr.values, o_s); rows[("sarima", "none")] = lag + zf
        except Exception as e:
            o_s = (0, 0, 0); rows[("sarima", "none")] = lag; logs.append((u, o, "sarima", "none", f"fallback {e}"))
        for (m, xs) in (VARIANTS if variants is None else variants):
            cols = xcols_of(xs)
            Xg = g[cols].copy()
            if Xg[cols].isna().all().any(): continue      # 이 동은 이 입력이 없음
            Xg = Xg.bfill().ffill()
            mu, sd = Xg[Xg.index <= o].mean(), Xg[Xg.index <= o].std().replace(0, 1)
            Xs = (Xg - mu) / sd
            try:
                if m == "arimax":
                    pred, cv = fit_pred(tr.values, o_a, Xs.reindex(tr.index).values, Xs.reindex(fut).values)
                else:
                    Xd = (Xs - Xs.shift(C.SEASON))
                    Xd_tr = Xd.reindex(ztr.index).fillna(0).values
                    zf, cv = fit_pred(ztr.values, o_s, Xd_tr, Xd.reindex(fut).fillna(0).values)
                    pred = lag + zf
                rows[(m, xs)] = pred
                if not cv: logs.append((u, o, m, xs, "not converged"))
            except Exception as e:
                rows[(m, xs)] = rows[("arima" if m == "arimax" else "sarima", "none")]; logs.append((u, o, m, xs, f"fallback {str(e)[:80]}"))
        for (m, xs), v in rows.items():
            fb = any(l[2] == m and l[3] == xs and str(l[4]).startswith("fallback") for l in logs if l[1] == o)
            out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "model": m, "xset": xs,
                                     "y_hat": np.expm1(np.asarray(v, dtype=float)), "fallback": int(fb)}))
        out.append(pd.DataFrame({"origin": [o], "unit": [u], "week_end": [pd.NaT], "h": [0], "model": ["_order"], "xset": [f"arima{o_a} sarima{o_s}"], "y_hat": [np.nan], "fallback": [0]}))
    return out, logs


if __name__ == "__main__":
    t0 = time.time()
    units = list(D.unit.unique())
    if len(sys.argv) > 1: units = units[: int(sys.argv[1])]   # 시험용: 앞의 N개 동만
    res = Parallel(n_jobs=-1, verbose=0)(delayed(run_unit)(u, D[D.unit == u]) for u in units)
    P = pd.concat([df for o, _ in res for df in o], ignore_index=True)
    L = pd.DataFrame([l for _, lg in res for l in lg], columns=["unit", "origin", "model", "xset", "msg"])
    orders = P[P.model == "_order"]; P = P[P.model != "_order"]
    for o, g in P.groupby("origin"):
        g.to_parquet(C.RESULTS / f"pred_{pd.Timestamp(o).date()}.parquet", index=False)
    orders.to_csv(C.RESULTS / "orders.csv", index=False)
    L.to_csv(C.RESULTS / "fit_log.csv", index=False)
    msg = f"units={len(units)}; rows={len(P)}; fallback={int(P.fallback.sum())}; sec={time.time() - t0:.0f}"
    log_run("02_models", msg); print(msg)
