"""Re-check the mobility-shape features on later origins, with data to 2025-08 (P.DATA_END) and day-level
anomaly correction (rule S4 from src/anomaly). Only features computable from the weekly series are used.

Origins: 13 future weeks contain no anomaly week (from 2025-04-28, every 2 weeks).
Truth: g = log share of the raw future 13 weeks - log share of the same weeks a year earlier (corrected).
Usage: python src/features_screen/run_feature_screen_extended.py <codex_root> <out_dir>
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
H, LAG = P.H, P.LAG
weeks, dongs, W, C = P.load_weekly_extended(ROOT)   # data to 2025-08 (P.DATA_END), C = day-level S4
n, N = W.shape
anom = P.anomaly_mask(weeks)
keep = P.size_floor(weeks, W)


def slope(Y):
    t = np.arange(len(Y)) - (len(Y) - 1) / 2
    return (t[:, None] * (Y - Y.mean(0))).sum(0) / (t ** 2).sum()


ls = np.log(C / C.sum(1, keepdims=True))
widx = {str(w): i for i, w in enumerate(weeks)}
origins = [o for o in range(widx["2025-04-28"], n - H, 2) if not anom[o + 1:o + 1 + H].any()]
res = {}
for o in origins:
    f = {"mob_g_13w (momentum)": P.g_at(C, o),
         "mob_g_4w": np.log(P.share(C, o - 3, o)) - np.log(P.share(C, o - 3 - LAG, o - LAG)),
         "mob_g_26w": np.log(P.share(C, o - 25, o)) - np.log(P.share(C, o - 25 - LAG, o - LAG)),
         "mob_share_slope_13w": slope(ls[o - 12:o + 1]),
         "mob_share_slope_26w": slope(ls[o - 25:o + 1]),
         "mob_volatility_26w": ls[o - 25:o + 1].std(0),
         "mob_log_size_13w": np.log(C[o - 12:o + 1].mean(0))}
    g = np.log(P.share(W, o + 1, o + H)) - np.log(P.share(C, o + 1 - LAG, o + H - LAG))
    mom = f["mob_g_13w (momentum)"]
    A_ = np.c_[np.ones(keep.sum()), mom[keep]]
    resid = g[keep] - A_ @ np.linalg.lstsq(A_, g[keep], rcond=None)[0]
    for k, v in f.items():
        r = res.setdefault(k, {"raw": [], "beyond": []})
        r["raw"].append(P.spearman(v[keep], g[keep]))
        if not k.startswith("mob_g_13w"):
            r["beyond"].append(P.spearman(v[keep], resid))
out = {k: {"spearman_with_future_change": round(float(np.mean(v["raw"])), 3),
           "spearman_beyond_momentum": round(float(np.mean(v["beyond"])), 3) if v["beyond"] else None,
           "beyond_min_max": [round(float(min(v["beyond"])), 3), round(float(max(v["beyond"])), 3)] if v["beyond"] else None}
       for k, v in res.items()}
json.dump({"origins": [str(weeks[o]) for o in origins], "dongs_above_floor": int(keep.sum()), "features": out},
          open(os.path.join(OUT, "screening_extended.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for k, v in out.items():
    print("%-26s %s" % (k, v))
