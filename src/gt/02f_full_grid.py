"""전체 경우의 수 실험 (2026-11-16): 팀원 설정(13주 주기·고정 차수·원래 지수) 재현과 수정 단계 비교
요인
  data   : raw(보정 전) / clean(2024-08~10 ÷ 1.52)
  fam    : ar  = 계절 없는 ARIMA
           s13 = SARIMA (p,d,q)(1,0,0)13  (팀원 형태: 13주 전 값에 계절 AR)
           s52 = 52주 계절 차분 위 ARMA = SARIMA (p,d,q)(0,1,0)52
  order  : fixed (ar (0,1,1), s13 (1,1,0), s52 (1,1,0)) / aic (p≤2, d≤1, q≤1 12개 중 AIC 최소, 입력 없는 모델에서 고르고 입력 모델에 그대로)
  search : none / idx(원래 지수 log1p) / share(서울 대비 비중 log)  — 4개월 전 달 값 (기존 규칙)
  lag13  : 0 / 1 (13주 전 생활이동 log1p를 입력으로, 팀원 입력)
  결합(작년 같은 주와 1:1)은 평가(03f)에서 계산
모든 모델 log1p(y)에 적합. 입력은 학습 구간으로 표준화. s52의 입력은 52주 차분.
출력: results/gt/grid/pred_<origin>.parquet  (column: data, fam, order, search, lag13, y_hat ...)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUT = C.RESULTS / ("grid_test" if len(sys.argv) > 1 else "grid"); OUT.mkdir(exist_ok=True)
GRID = [(p, d, q) for p in range(3) for d in range(2) for q in range(2)]
FIXED = {"ar": (0, 1, 1), "s13": (1, 1, 0), "s52": (1, 1, 0)}
SEAS = {"ar": (0, 0, 0, 0), "s13": (1, 0, 0, 13), "s52": (0, 0, 0, 0)}

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
fac = pd.read_csv(C.RESULTS / "clean_factor.csv", index_col=0, parse_dates=True).factor
T["yc"] = T.y / T.week_end.map(fac).fillna(1.0)
X = pd.read_parquet(C.INTERIM / "features_weekly.parquet")[["unit", "week_end", "search_share", "search_index"]]
D = X.merge(T, on=["unit", "week_end"], how="left").sort_values(["unit", "week_end"])


def fit(z, order, fam, ex=None):
    return SARIMAX(z, exog=ex, order=order, seasonal_order=SEAS[fam], trend="c" if order[1] == 0 else "n").fit(disp=False, maxiter=200)


def aic_order(z, fam):
    best = None
    for o in GRID:
        try:
            r = fit(z, o, fam)
            if np.isfinite(r.aic) and (best is None or r.aic < best[0]): best = (r.aic, o)
        except Exception:
            pass
    return best[1] if best else FIXED[fam]


def run_unit(u, g):
    g = g.set_index("week_end"); out = []
    S = g[["search_index", "search_share"]].bfill().ffill()
    for o in pd.to_datetime(C.ORIGINS):
        fut = g.index[g.index > o][:C.H]
        if len(fut) < C.H: continue
        for data, col in (("raw", "y"), ("clean", "yc")):
            ly = np.log1p(g[col])
            tr = ly[ly.index <= o].dropna()
            if len(tr) < 60: continue
            lag52 = ly.reindex(fut - pd.Timedelta(weeks=C.SEASON)).values
            out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "data": data, "fam": "snaive", "order": "-", "search": "none", "lag13": 0, "y_hat": np.expm1(lag52)}))
            L13 = ly.shift(13).bfill()
            for fam in ("ar", "s13", "s52"):
                if fam == "s52":
                    z = (ly - ly.shift(C.SEASON)); ztr = z[z.index <= o].dropna()
                    if len(ztr) < 30: continue
                    zidx = ztr.index
                else:
                    ztr, zidx = tr, tr.index
                orders = {"fixed": FIXED[fam], "aic": aic_order(ztr.values, fam)}
                for oname, od in orders.items():
                    for srch in ("none", "idx", "share"):
                        for l13 in (0, 1):
                            cols = ([] if srch == "none" else [S["search_index" if srch == "idx" else "search_share"]]) + ([L13] if l13 else [])
                            ex_tr = ex_fu = None
                            if cols:
                                Xg = pd.concat(cols, axis=1)
                                mu, sd = Xg[Xg.index <= o].mean(), Xg[Xg.index <= o].std().replace(0, 1)
                                Xs = (Xg - mu) / sd
                                if fam == "s52": Xs = (Xs - Xs.shift(C.SEASON))
                                ex_tr = Xs.reindex(zidx).fillna(0).values; ex_fu = Xs.reindex(fut).fillna(0).values
                            try:
                                r = fit(ztr.values, od, fam, ex_tr)
                                pr = np.asarray(r.get_forecast(C.H, exog=ex_fu).predicted_mean)
                                if fam == "s52": pr = lag52 + pr
                            except Exception:
                                continue
                            out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "data": data, "fam": fam,
                                                     "order": oname, "search": srch, "lag13": l13, "y_hat": np.expm1(pr), "ord": str(od)}))
    return out


if __name__ == "__main__":
    t0 = time.time()
    units = list(D.unit.unique())
    if len(sys.argv) > 1: units = units[: int(sys.argv[1])]
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(run_unit)(u, D[D.unit == u]) for u in units)
    P = pd.concat([d for r in res for d in r], ignore_index=True)
    for o, gg in P.groupby("origin"): gg.to_parquet(OUT / f"pred_{pd.Timestamp(o).date()}.parquet", index=False)
    msg = f"units={len(units)}; rows={len(P)}; sec={time.time() - t0:.0f}"
    log_run("02f_full_grid", msg); print(msg)
