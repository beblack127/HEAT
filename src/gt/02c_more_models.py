"""추가 모델 실험 (2026-11-16): 자료 이상 처리 + 새 모델
1) 자료 이상 처리 (2024-08-01 ~ 2024-10-31, 서울 합계 그래프로 정한 구간)
   - 서울 합계의 '기대값' = 작년 같은 주 서울 합계 x (2024년 1~7월 / 2023년 1~7월 서울 합계 비율). 과거 값만 쓴다
   - 배율 = 구간 서울 합계 평균 / 기대값 평균 (구간 전체에 하나). 이 구간의 모든 동 값을 배율로 나눈다 (동 사이 상대 크기는 유지)
   - 학습 자료와 '작년 같은 주' 값에 보정 값을 쓴다. 평가는 원래 값 기준 (보고서에 보정 값 기준도 함께)
2) 모델 (모두 log1p(y)에 적합, 동마다, 4개 시점, 13주 앞)
   - snaive_c, arima_c, sarima_c : 기존 모델을 보정 자료로 다시
   - arima_fourier : ARIMA + 1년 주기 푸리에 항 K=3쌍 (차수는 같은 12개 후보 중 AIC)
   - stl_ets       : STL(주기 52, robust)로 계절을 떼고 감쇠 추세 ETS로 예측한 뒤 계절을 더함
   - theta         : Theta (주기 52 계절 조정)
   결합은 03c에서 계산한다.
출력: results/gt/pred_more_<origin>.parquet, results/gt/clean_factor.csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

A0, A1 = pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-31")
K_FOURIER, PERIOD = 3, 365.25 / 7
GRID = [(p, d, q) for p in range(3) for d in range(2) for q in range(2)]

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y", "seoul_y"]].sort_values(["unit", "week_end"])
s = T.groupby("week_end").seoul_y.first()
r = s["2024-01-01":"2024-07-31"].mean() / s["2023-01-01":"2023-07-31"].mean()
exp_ = s.shift(52) * r
win = (s.index >= A0) & (s.index <= A1)
F0 = s[win].mean() / exp_[win].mean()          # 구간 전체에 하나의 배율 (주마다 다르게 하면 작년 명절 위치 차이까지 섞인다)
fac = pd.Series(np.where(win, F0, 1.0), index=s.index)
fac.rename("factor").to_csv(C.RESULTS / "clean_factor.csv")
T["yc"] = T.y / T.week_end.map(fac)


def fourier(idx):
    t = np.asarray((idx - pd.Timestamp("2023-01-01")).days / 7.0)
    return np.column_stack([f(2 * np.pi * k * t / PERIOD) for k in range(1, K_FOURIER + 1) for f in (np.sin, np.cos)])


def fit_best(z, exog=None):
    best = None
    for (p, d, q) in GRID:
        try:
            r_ = SARIMAX(z, exog=exog, order=(p, d, q), trend="c" if d == 0 else "n").fit(disp=False, maxiter=200)
            if np.isfinite(r_.aic) and (best is None or r_.aic < best[0]): best = (r_.aic, (p, d, q), r_)
        except Exception:
            pass
    return best


def run_unit(u, g):
    g = g.set_index("week_end")
    ly, lyc = np.log1p(g.y), np.log1p(g.yc)
    out, logs = [], []
    for o in pd.to_datetime(C.ORIGINS):
        tr = lyc[lyc.index <= o].dropna()
        idx_all = g.index
        fut = idx_all[idx_all > o][:C.H]
        if len(fut) < C.H:   # 마지막 자료 뒤는 주 달력으로 만든다
            fut = pd.date_range(o + pd.Timedelta(weeks=1), periods=C.H, freq="W-SUN")
        if len(tr) < 60: continue
        rows = {}
        lag = lyc.reindex(fut - pd.Timedelta(weeks=C.SEASON)).values
        rows["snaive_c"] = lag
        b = fit_best(tr.values)
        if b: rows["arima_c"] = np.asarray(b[2].get_forecast(C.H).predicted_mean)
        z = (lyc - lyc.shift(C.SEASON)); ztr = z[z.index <= o].dropna()
        bs = fit_best(ztr.values) if len(ztr) >= 30 else None
        rows["sarima_c"] = lag + (np.asarray(bs[2].get_forecast(C.H).predicted_mean) if bs else 0)
        bf = fit_best(tr.values, fourier(tr.index))
        if bf: rows["arima_fourier"] = np.asarray(bf[2].get_forecast(C.H, exog=fourier(fut)).predicted_mean)
        try:
            rows["stl_ets"] = np.asarray(STLForecast(tr.values, ExponentialSmoothing, model_kwargs=dict(trend="add", damped_trend=True),
                                                     period=C.SEASON, robust=True).fit().forecast(C.H))
        except Exception as e: logs.append((u, o, "stl_ets", str(e)[:80]))
        try:
            rows["theta"] = np.asarray(ThetaModel(tr.values, period=C.SEASON, deseasonalize=True, method="additive").fit().forecast(C.H))
        except Exception as e: logs.append((u, o, "theta", str(e)[:80]))
        for k, v in rows.items():
            out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "model": k, "xset": "none",
                                     "y_hat": np.expm1(np.asarray(v, dtype=float))}))
    return out, logs


if __name__ == "__main__":
    t0 = time.time()
    units = list(T.unit.unique())
    if len(sys.argv) > 1: units = units[: int(sys.argv[1])]
    res = Parallel(n_jobs=-1)(delayed(run_unit)(u, T[T.unit == u]) for u in units)
    P = pd.concat([d for o, _ in res for d in o], ignore_index=True)
    L = [l for _, lg in res for l in lg]
    for o, gg in P.groupby("origin"):
        gg.to_parquet(C.RESULTS / f"pred_more_{pd.Timestamp(o).date()}.parquet", index=False)
    msg = f"units={len(units)}; rows={len(P)}; errors={len(L)}; factor_max={fac.max():.3f}; sec={time.time() - t0:.0f}"
    log_run("02c_more_models", msg); print(msg)
