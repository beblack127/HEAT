"""Next checks after the combo model (reports/combo):
 (a) neighbor signal: does recent growth of nearby dongs / the same gu tell us more than the dong's own?
 (b) light nonlinear model: gradient-boosted trees on the same features, fixed settings (no tuning)
 (c) where predictions fail: hit rate of the true emerging dongs by dong type

Setup as in reports/combo: weekly 20-39, purposes 4+5+7, data to 2025-08-31 (P.DATA_END), day-level S4,
dongs above the size floor. Target for window h: change in Seoul share vs the same weeks a year earlier.
Test origins: every week from 2025-04-28 whose h future weeks are clean (h=4: 14, h=13: 5). Models are
refit at each test origin on origins whose target window has ended (o' + h <= o).
Neighbors: the 6 nearest dongs by centroid (01_행정동_목록.csv lat/lon); neighbor and gu signals use the
summed counts of the group (not a mean of shares), so small dongs do not dominate.
Usage: python src/neighbors/run_neighbor_nonlinear.py <codex_root> <out_dir>
"""
import csv, json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LAG, K_NB = P.LAG, 6
weeks, dongs, W, C = P.load_weekly_extended(ROOT)
n, N = W.shape
anom = P.anomaly_mask(weeks)
keep = P.size_floor(weeks, W)
widx = {str(w): i for i, w in enumerate(weeks)}
ls = np.log(C / C.sum(1, keepdims=True))
O_MIN = 12 + LAG

# ---------- groups: neighbors and gu ----------
info = {}
with open(os.path.join(P.HEAT, "data", "processed", "nametags", "01_행정동_목록.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        info[r["dong_code8"]] = (float(r["lat"]), float(r["lon"]), r["gu"], r["dong_name"])
gu_of = np.array([c[:5] for c in dongs])
has = np.array([c in info for c in dongs])
lat = np.array([info[c][0] if c in info else np.nan for c in dongs])
lon = np.array([info[c][1] if c in info else np.nan for c in dongs])
NB = np.zeros((N, N))           # row i: 1 for the K_NB nearest dongs (dongs without coords: same gu)
for i in range(N):
    if has[i]:
        d = np.hypot((lat - lat[i]) * 111.0, (lon - lon[i]) * 88.2)
        d[i] = np.inf; d[~has] = np.inf
        NB[i, np.argsort(d)[:K_NB]] = 1
    else:
        NB[i, (gu_of == gu_of[i]) & (np.arange(N) != i)] = 1
GU = (gu_of[:, None] == gu_of[None, :]).astype(float)
np.fill_diagonal(GU, 0)


def slope(Y):
    t = np.arange(len(Y)) - (len(Y) - 1) / 2
    return (t[:, None] * (Y - Y.mean(0))).sum(0) / (t ** 2).sum()


def gwin(X, o_end, h):
    """share change of the h weeks ending at o_end vs a year earlier, for each column of X (counts)"""
    a = X[o_end - h + 1:o_end + 1].sum(0); b = X[o_end - h + 1 - LAG:o_end + 1 - LAG].sum(0)
    return np.log(a / a.sum()) - np.log(b / b.sum())


def group_g(M, o_end, h):
    """share change of each dong's group (sum of the group's counts)"""
    a = C[o_end - h + 1:o_end + 1].sum(0) @ M.T; b = C[o_end - h + 1 - LAG:o_end + 1 - LAG].sum(0) @ M.T
    tot_a, tot_b = C[o_end - h + 1:o_end + 1].sum(), C[o_end - h + 1 - LAG:o_end + 1 - LAG].sum()
    return np.log(a / tot_a) - np.log(b / tot_b)


OWN = ["momentum_13w", "change_4w", "slope_13w", "slope_26w", "log_size", "volatility_26w"]
NBF = ["nb_change_4w", "nb_momentum_13w", "gu_change_4w"]
FEATS = OWN + NBF


def feats(o):
    X = np.c_[gwin(C, o, 13), gwin(C, o, 4), slope(ls[o - 12:o + 1]), slope(ls[o - 25:o + 1]),
              np.log(C[o - 12:o + 1].mean(0)), ls[o - 25:o + 1].std(0),
              group_g(NB, o, 4), group_g(NB, o, 13), group_g(GU, o, 4)][keep]
    return (X - X.mean(0)) / X.std(0)


z = lambda v: (v - v.mean()) / v.std()
Xo = {o: feats(o) for o in range(O_MIN, n)}
fi = {f: FEATS.index(f) for f in FEATS}
MODELS = {
    "momentum": ("single", ["momentum_13w"]),
    "change_4w": ("single", ["change_4w"]),
    "linear own (4)": ("ols", ["momentum_13w", "change_4w", "slope_13w", "slope_26w"]),
    "linear own + neighbor": ("ols", ["momentum_13w", "change_4w", "slope_13w", "slope_26w"] + NBF),
    "trees own (6)": ("hgb", OWN),
    "trees own + neighbor": ("hgb", FEATS),
}
HGB = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=42)

rows, beyond, preds, coefs = [], [], [], {}
for h in (4, 8, 13):
    tests = [o for o in range(widx["2025-04-28"], n - h) if not anom[o + 1:o + 1 + h].any()]
    for o in tests:
        g = gwin(C, o + h, h)[keep]
        top_true = P.topk(g)
        # diagnostic: neighbor features vs what own change_4w + momentum leave unexplained (uses the truth)
        A_ = np.c_[np.ones(len(g)), Xo[o][:, [fi["change_4w"], fi["momentum_13w"]]]]
        res = g - A_ @ np.linalg.lstsq(A_, g, rcond=None)[0]
        for f in NBF + ["slope_13w", "log_size"]:
            beyond.append({"h": h, "origin": str(weeks[o]), "feature": f, "spearman_beyond_own": P.spearman(Xo[o][:, fi[f]], res)})
        train = list(range(O_MIN, o - h + 1))
        ytr = np.concatenate([z(gwin(C, p + h, h)[keep]) for p in train])
        for m, (kind, cols) in MODELS.items():
            ci = [fi[c] for c in cols]
            if kind == "single":
                pred = Xo[o][:, ci[0]]
            else:
                Atr = np.vstack([Xo[p][:, ci] for p in train])
                if kind == "ols":
                    b = np.linalg.lstsq(Atr, ytr, rcond=None)[0]; pred = Xo[o][:, ci] @ b
                    coefs.setdefault((h, m), []).append(b)
                else:
                    pred = HistGradientBoostingRegressor(**HGB).fit(Atr, ytr).predict(Xo[o][:, ci])
            top_pred = P.topk(pred)
            rows.append({"h": h, "origin": str(weeks[o]), "model": m, "spearman": P.spearman(pred, g),
                         "emerging_P@10": len(top_pred & top_true) / len(top_true)})
            preds.append((h, o, m, pred, g))

# ---------- (c) failure by dong type ----------
kd = [c for c, k in zip(dongs, keep) if k]
Wk = W[:, keep]
y23 = [i for i, w in enumerate(weeks) if w.year == 2023]
size_q = np.digitize(np.log(Wk[y23].mean(0)), np.quantile(np.log(Wk[y23].mean(0)), [0.25, 0.5, 0.75]))
dates, dd, Yd, _ = P.load_daily(ROOT)
_, Wwe = P.weekly(dates, Yd, (5, 6)); _, Wwd = P.weekly(dates, Yd, range(5))
jj = [dd.index(c) for c in kd]
wk_ratio = Wwe[:52, jj].mean(0) / Wwd[:52, jj].mean(0)          # 2023 weekend / weekday mean
wr_t = np.digitize(wk_ratio, np.quantile(wk_ratio, [1 / 3, 2 / 3]))
ref121, careet = P.ref_121(), set(P.CAREET) | set(P.SEONGSU)
TYPES = {
    "size: bottom quarter (of kept)": size_q == 0, "size: 2nd": size_q == 1, "size: 3rd": size_q == 2, "size: top quarter": size_q == 3,
    "weekday-heavy (office-like, low weekend/weekday third)": wr_t == 0, "middle third": wr_t == 1,
    "weekend-heavy (leisure-like, top third)": wr_t == 2,
    "Seoul 121 places (developed commercial / tourist)": np.array([c in ref121 for c in kd]),
    "Careet Gen-Z list + Seongsu": np.array([c in careet for c in kd]),
}
fail = []
for h in (4, 13):
    for m in ("change_4w", "trees own + neighbor"):
        P_ = [(pred, g) for (hh, o, mm, pred, g) in preds if hh == h and mm == m]
        for t, mask in TYPES.items():
            tp = fn = fp = 0; rank_err = []
            for pred, g in P_:
                tt, tpd = P.topk(g), P.topk(pred)
                idx = set(np.where(mask)[0])
                tp += len(tt & tpd & idx); fn += len((tt - tpd) & idx); fp += len((tpd - tt) & idx)
                rp, rg = P.rank(pred) / (len(g) - 1), P.rank(g) / (len(g) - 1)
                rank_err.append(np.abs(rp - rg)[mask].mean())
            fail.append({"h": h, "model": m, "type": t, "dongs": int(mask.sum()), "true_emerging_dong_weeks": tp + fn,
                         "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                         "false_alarms": fp, "mean_rank_error": round(float(np.mean(rank_err)), 3)})
# dongs most often missed (true emerging, not predicted) at h=13 by the best simple rule
nm = P.names()
miss = {}
for (hh, o, mm, pred, g) in preds:
    if hh == 13 and mm == "change_4w":
        for i in P.topk(g) - P.topk(pred):
            miss[kd[i]] = miss.get(kd[i], 0) + 1

# ---------- outputs ----------
with open(os.path.join(OUT, "scores_by_origin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
with open(os.path.join(OUT, "failure_by_type.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(fail[0])); w.writeheader(); w.writerows(fail)
summary = {"dongs": int(keep.sum()), "neighbors_k": K_NB, "dongs_without_coords": [c for c, k in zip(dongs, has) if not k],
           "by_h": {}, "beyond_own": {}, "failure_by_type": fail,
           "most_missed_h13_change_4w": sorted([(nm.get(c, c), k) for c, k in miss.items()], key=lambda x: -x[1])[:15]}
for h in (4, 8, 13):
    hs = [r for r in rows if r["h"] == h]
    summary["by_h"][h] = {"test_origins": len({r["origin"] for r in hs}),
                          **{m: {"emerging_P@10": round(float(np.mean([r["emerging_P@10"] for r in hs if r["model"] == m])), 3),
                                 "spearman": round(float(np.mean([r["spearman"] for r in hs if r["model"] == m])), 3)} for m in MODELS},
                          "linear own + neighbor coef": dict(zip(MODELS["linear own + neighbor"][1], np.round(np.mean(coefs[(h, "linear own + neighbor")], 0), 3).tolist()))}
    bs = [b for b in beyond if b["h"] == h]
    summary["beyond_own"][h] = {f: {"mean": round(float(np.mean([b["spearman_beyond_own"] for b in bs if b["feature"] == f])), 3),
                                    "min_max": [round(float(min(b["spearman_beyond_own"] for b in bs if b["feature"] == f)), 3),
                                                round(float(max(b["spearman_beyond_own"] for b in bs if b["feature"] == f)), 3)]}
                                for f in NBF + ["slope_13w", "log_size"]}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "failure_by_type"}, ensure_ascii=False, indent=1))
for r in fail:
    print(r)
