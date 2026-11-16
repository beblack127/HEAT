"""팀원 설정(고정 차수, 보정 전)에서 '아무 입력이나 좋아지나' 확인 (2026-11-16)
fam ar (0,1,1), s13 (1,1,0)(1,0,0)13, s52 (1,1,0) on 52주 차분, 모두 13주 전 생활이동 입력 포함(팀원과 같게)
입력: none / search_idx(원래 지수) / placebo(다른 동의 원래 검색 지수, 동 순서를 seed 42로 섞음) / subway / fnb
출력: results/gt/grid/placebo_<origin>.parquet
"""
import sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config as C
from common import log_run

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

FIXED = {"ar": (0, 1, 1), "s13": (1, 1, 0), "s52": (1, 1, 0)}
SEAS = {"ar": (0, 0, 0, 0), "s13": (1, 0, 0, 13), "s52": (0, 0, 0, 0)}
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")[["unit", "week_end", "y"]]
X = pd.read_parquet(C.INTERIM / "features_weekly.parquet")[["unit", "week_end", "search_index", "subway_exit_4w", "fnb_net3"]]
D = X.merge(T, on=["unit", "week_end"], how="left").sort_values(["unit", "week_end"])
units = list(D.unit.unique())
perm = dict(zip(units, np.random.default_rng(C.SEED).permutation(units)))
PL = D.pivot(index="week_end", columns="unit", values="search_index")


def run_unit(u, g):
    g = g.set_index("week_end"); out = []
    g["placebo"] = PL[perm[u]].reindex(g.index)
    ly = np.log1p(g.y); L13 = ly.shift(13).bfill()
    for o in pd.to_datetime(C.ORIGINS):
        fut = g.index[g.index > o][:C.H]
        tr = ly[ly.index <= o].dropna()
        if len(fut) < C.H or len(tr) < 60: continue
        lag52 = ly.reindex(fut - pd.Timedelta(weeks=C.SEASON)).values
        for fam in ("ar", "s13", "s52"):
            if fam == "s52":
                z = ly - ly.shift(C.SEASON); ztr = z[z.index <= o].dropna()
            else:
                ztr = tr
            for inp in ("none", "search_index", "placebo", "subway_exit_4w", "fnb_net3"):
                cols = [L13] + ([g[inp]] if inp != "none" else [])
                Xg = pd.concat(cols, axis=1).bfill().ffill()
                if Xg.isna().all().any(): continue
                mu, sd = Xg[Xg.index <= o].mean(), Xg[Xg.index <= o].std().replace(0, 1)
                Xs = (Xg - mu) / sd
                if fam == "s52": Xs = Xs - Xs.shift(C.SEASON)
                try:
                    od = FIXED[fam]
                    r = SARIMAX(ztr.values, exog=Xs.reindex(ztr.index).fillna(0).values, order=od, seasonal_order=SEAS[fam], trend="n").fit(disp=False, maxiter=200)
                    pr = np.asarray(r.get_forecast(C.H, exog=Xs.reindex(fut).fillna(0).values).predicted_mean)
                    if fam == "s52": pr = lag52 + pr
                except Exception:
                    continue
                out.append(pd.DataFrame({"origin": o, "unit": u, "week_end": fut, "h": np.arange(1, C.H + 1), "fam": fam, "input": inp, "y_hat": np.expm1(pr)}))
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(run_unit)(u, D[D.unit == u]) for u in units)
    P = pd.concat([d for r in res for d in r], ignore_index=True)
    (C.RESULTS / "grid").mkdir(exist_ok=True)
    for o, gg in P.groupby("origin"): gg.to_parquet(C.RESULTS / "grid" / f"placebo_{pd.Timestamp(o).date()}.parquet", index=False)
    msg = f"rows={len(P)}; sec={time.time() - t0:.0f}"; log_run("02g_placebo", msg); print(msg)
