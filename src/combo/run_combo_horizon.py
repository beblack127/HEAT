"""(2) Does a light linear combination of mobility-shape features beat momentum alone?
(3) How does predictability change with the forecast window (4 / 8 / 13 / 26 weeks)?

Target for window h at origin o: g_h = log Seoul share of weeks o+1..o+h minus that of the same weeks one
year earlier (20-39, purposes 4+5+7, history corrected with rule S4, dongs above the size floor).
Features at o (all from the weekly series): momentum (13-week share change vs a year earlier), 4-week
change, share slope over 13 and 26 weeks, log size (13-week mean).
Combination: pooled least squares on per-origin z-scores, refit at every test origin on training origins
whose target window has ended by the test origin (o' + h <= o). Training origins start at 2024-03-25 (the
first with a full feature history) and include corrected anomaly weeks.
Test origins: every week from 2025-04-28 whose h future weeks contain no anomaly week and end by 2025-12-22.
Metrics: rank correlation with g_h, and emerging P@10 (overlap of predicted and true top 10%).
Usage: python src/combo/run_combo_horizon.py <codex_root> <out_dir>
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LAG = P.LAG
weeks, dongs, W = P.load_weekly_extended(ROOT)
n, N = W.shape
C = P.correct_s4(weeks, W)
anom = P.anomaly_mask(weeks)
keep = P.size_floor(weeks, W)
ls = np.log(C / C.sum(1, keepdims=True))
widx = {str(w): i for i, w in enumerate(weeks)}
FEATS = ["momentum_13w", "change_4w", "slope_13w", "slope_26w", "log_size"]
O_MIN = 12 + LAG  # first origin with a full 13-week momentum history


def slope(Y):
    t = np.arange(len(Y)) - (len(Y) - 1) / 2
    return (t[:, None] * (Y - Y.mean(0))).sum(0) / (t ** 2).sum()


def gwin(o_end, h):
    return np.log(P.share(C, o_end - h + 1, o_end)) - np.log(P.share(C, o_end - h + 1 - LAG, o_end - LAG))


def feats(o):
    X = np.c_[gwin(o, 13), gwin(o, 4), slope(ls[o - 12:o + 1]), slope(ls[o - 25:o + 1]),
              np.log(C[o - 12:o + 1].mean(0))][keep]
    return (X - X.mean(0)) / X.std(0)


def z(v):
    return (v - v.mean()) / v.std()


Xo = {o: feats(o) for o in range(O_MIN, n)}
MODELS = {"momentum": None, "change_4w": None, "combo (5 features)": FEATS, "combo without size": FEATS[:4]}
rows, coefs = [], {}
HS = [h for h in (4, 8, 13, 26) if any(not anom[o + 1:o + 1 + h].any() for o in range(widx["2025-04-28"], n - h))]
for h in HS:
    tests = [o for o in range(widx["2025-04-28"], n - h) if not anom[o + 1:o + 1 + h].any()]
    for o in tests:
        g = gwin(o + h, h)[keep]
        top_true = P.topk(g)
        train = [p for p in range(O_MIN, o - h + 1)]
        for m, cols in MODELS.items():
            if cols is None:
                pred = Xo[o][:, FEATS.index("momentum_13w" if m == "momentum" else "change_4w")]
            else:
                ci = [FEATS.index(c) for c in cols]
                A = np.vstack([Xo[p][:, ci] for p in train])
                y = np.concatenate([z(gwin(p + h, h)[keep]) for p in train])
                b = np.linalg.lstsq(A, y, rcond=None)[0]
                pred = Xo[o][:, ci] @ b
                if m == "combo (5 features)":
                    coefs.setdefault(h, []).append(b)
            rows.append({"h": h, "origin": str(weeks[o]), "model": m, "train_origins": len(train),
                         "spearman": P.spearman(pred, g), "emerging_P@10": len(P.topk(pred) & top_true) / len(top_true)})

with open(os.path.join(OUT, "scores_by_origin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

common = set.intersection(*[{r["origin"] for r in rows if r["h"] == h} for h in HS])
summary = {"dongs": int(keep.sum()), "common_origins": sorted(common), "by_h": {}}
for h in HS:
    hs = [r for r in rows if r["h"] == h]
    origins = sorted({r["origin"] for r in hs})
    d = {"test_origins": len(origins), "first_last": [origins[0], origins[-1]],
         "train_origins_range": [min(r["train_origins"] for r in hs), max(r["train_origins"] for r in hs)],
         "combo_coef_mean": dict(zip(FEATS, np.round(np.mean(coefs[h], 0), 3).tolist()))}
    for m in MODELS:
        ms = [r for r in hs if r["model"] == m]
        mc = [r for r in ms if r["origin"] in common]
        d[m] = {"spearman": round(float(np.mean([r["spearman"] for r in ms])), 3),
                "emerging_P@10": round(float(np.mean([r["emerging_P@10"] for r in ms])), 3),
                "common_spearman": round(float(np.mean([r["spearman"] for r in mc])), 3),
                "common_P@10": round(float(np.mean([r["emerging_P@10"] for r in mc])), 3)}
    mom = {r["origin"]: r for r in hs if r["model"] == "momentum"}
    cmb = [r for r in hs if r["model"] == "combo (5 features)"]
    d["combo_beats_momentum"] = {"spearman": sum(r["spearman"] > mom[r["origin"]]["spearman"] for r in cmb),
                                 "P@10_better_equal_worse": [sum(r["emerging_P@10"] > mom[r["origin"]]["emerging_P@10"] for r in cmb),
                                                              sum(r["emerging_P@10"] == mom[r["origin"]]["emerging_P@10"] for r in cmb),
                                                              sum(r["emerging_P@10"] < mom[r["origin"]]["emerging_P@10"] for r in cmb)],
                                 "of": len(cmb)}
    summary["by_h"][h] = d
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
