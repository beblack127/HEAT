"""Light ground-truth sensitivity: does the conclusion change if the target is built a little differently?

Variants (all from the same daily file, purposes 4+5+7):
  V0 20-39, all days (current target)     V1 all ages, all days
  V2 20-39, weekends only                 V3 20-39, weekdays only
  V4 20-39, one-off spikes removed (src/peaks rule, replaced by the same-weekday baseline)
Checks: (a) current hotspots, 2025 Jan-Aug: overlap with V0, recall/precision of known hot places
(Seoul 121 places: developed commercial areas + tourist zones; Careet Gen-Z list + Seongsu);
(b) emerging: agreement of the share change g with V0, momentum predictability, and where known
rising dongs land.
Usage: python src/gt_refine/run_gt_sensitivity.py <codex_root> <out_dir>
"""
import json, os, sys
from datetime import date, timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
H, LAG = P.H, P.LAG
dates, dongs, Y, A = P.load_daily(ROOT)
nm = P.names()
ci = {c: j for j, c in enumerate(dongs)}

# spikes removed (same rule as src/peaks/run_peaks.py)
day_i = {d: i for i, d in enumerate(dates)}
Yc = Y.copy()
flag = np.zeros_like(Y, dtype=bool); base = np.full_like(Y, np.nan)
for t in range(len(dates)):
    nb = [day_i.get(dates[t] + timedelta(weeks=k)) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    nb = [i for i in nb if i is not None]
    if len(nb) < 6:
        continue
    v = Y[nb]; med = np.median(v, axis=0); sc = np.maximum(1.4826 * np.median(np.abs(v - med), axis=0), 0.05 * med)
    flag[t] = (Y[t] >= 1.5 * med) & ((Y[t] - med) / sc >= 5); base[t] = med
Yc[flag] = base[flag]

weeks, W0 = P.weekly(dates, Y)
V = {"V0 20-39 all days": W0,
     "V1 all ages": P.weekly(dates, A)[1],
     "V2 20-39 weekends": P.weekly(dates, Y, days_of_week=(5, 6))[1],
     "V3 20-39 weekdays": P.weekly(dates, Y, days_of_week=(0, 1, 2, 3, 4))[1],
     "V4 20-39 spikes removed": P.weekly(dates, Yc)[1]}
keep = P.size_floor(weeks, W0)
anom = P.anomaly_rows(weeks)
ref121 = {ci[c] for c in P.ref_121() if c in ci}
refC = {ci[c] for c in list(P.CAREET) + list(P.SEONGSU) if c in ci}
rising = {"11140615": "신당동", "11170625": "한강로동(용리단길)", "11410615": "연희동", "11200670": "성수2가1동",
          "11200690": "성수2가3동", "11170685": "한남동", "11170520": "용산2가동(해방촌)", "11170660": "이태원2동(경리단길)"}

# (a) current hotspots, 2025
rows25 = [i for i, w in enumerate(weeks) if w.year == 2025]
lvl = {k: W[rows25].mean(axis=0) for k, W in V.items()}
top0 = P.topk(lvl["V0 20-39 all days"])
cur = {}
for k, L in lvl.items():
    out = {}
    for q in (0.05, 0.10, 0.20):
        t = P.topk(L, q)
        out[f"top{int(q * 100)}"] = {"recall_121": round(len(t & ref121) / len(ref121), 3), "precision_121": round(len(t & ref121) / len(t), 3),
                                    "recall_careet": round(len(t & refC) / len(refC), 3)}
    out["overlap_top10_with_V0"] = round(len(P.topk(L) & top0) / len(top0), 3)
    out["rank_corr_with_V0"] = round(P.spearman(L, lvl["V0 20-39 all days"]), 3)
    out["top10_not_in_V0"] = [nm.get(dongs[j], dongs[j]) for j in sorted(P.topk(L) - top0, key=lambda j: -L[j])]
    cur[k] = out

# (b) emerging
origins = list(range(LAG + 2 * H - 1, len(weeks) - H, 4))
clean = []
for o in origins:
    win = [(o + 1, o + H), (o + 1 - LAG, o + H - LAG), (o - H + 1, o), (o - H + 1 - LAG, o - LAG)]
    clean.append(not any(anom[a:b + 1].any() for a, b in win))
emer = {}
for k, W in V.items():
    agree, ov, mom_s, mom_p, pct = [], [], [], [], {c: [] for c in rising}
    for o, cl in zip(origins, clean):
        g = (np.log(P.share(W, o + 1, o + H)) - np.log(P.share(W, o + 1 - LAG, o + H - LAG)))[keep]
        g0 = (np.log(P.share(W0, o + 1, o + H)) - np.log(P.share(W0, o + 1 - LAG, o + H - LAG)))[keep]
        gm = P.g_at(W, o)[keep]
        agree.append(P.spearman(g, g0)); ov.append(len(P.topk(g) & P.topk(g0)) / len(P.topk(g0)))
        mom_s.append(P.spearman(gm, g)); mom_p.append(len(P.topk(gm) & P.topk(g)) / len(P.topk(g)))
        if cl:
            kept = np.nonzero(keep)[0]
            r = P.rank(g) / (len(g) - 1)
            for c in rising:
                if c in ci and ci[c] in set(kept):
                    pct[c].append(float(r[list(kept).index(ci[c])]))
    emer[k] = {"g_rank_corr_with_V0": round(float(np.mean(agree)), 3), "top10_overlap_with_V0": round(float(np.mean(ov)), 3),
               "momentum_rank_corr": round(float(np.mean(mom_s)), 3), "momentum_P@10": round(float(np.mean(mom_p)), 3),
               "known_rising_percentile_clean_origins": {rising[c]: (round(float(np.mean(v)), 2) if v else None) for c, v in pct.items()}}

summary = {"dongs": len(dongs), "dongs_above_floor": int(keep.sum()), "ref_121_dongs": len(ref121), "ref_careet_dongs": len(refC),
           "origins": len(origins), "clean_origins": int(sum(clean)), "current_hotspots_2025": cur, "emerging": emer}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
