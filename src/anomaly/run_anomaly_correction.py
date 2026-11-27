"""How should the three anomaly windows be corrected? Compare correction rules by how well the corrected
history forecasts clean future weeks.

Anomaly windows (src/gt_refine/panel.py ANOMALIES): 2024-08~10, 2025-01, 2025-03~04.
Rules (each window is divided by a factor; other weeks unchanged):
  S0 raw                    no correction
  S1 midterm               2024-08~10 only, all dongs / 1.52
  S2 Seoul factor          one factor per window for all dongs
  S3 dong factor           one factor per window and dong
  S4 dong factor, shrunk   S3 pulled halfway (in logs) toward S2
Factor = observed / expected over the window, expected = same weeks one year earlier x the YoY ratio
of the (up to) 8 clean weeks just before the window. Windows are corrected in time order, so later
windows use already-corrected history.

Test: origins whose 13 future weeks contain no anomaly week but whose history does (2025-04-28 ..
2025-09-22, every 2 weeks). The truth is the raw (clean) future. Baselines are built from the corrected
history. If a rule is right, last-year-based forecasts of clean weeks should improve.
Usage: python src/anomaly/run_anomaly_correction.py <codex_root> <out_dir>
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
H, LAG = P.H, P.LAG
weeks, dongs, W = P.load_weekly_extended(ROOT)
n, N = W.shape
anom_all = P.anomaly_mask(weeks)
keep = P.size_floor(weeks, W)


def window_rows(k):
    a, b = P.ANOMALIES[k]
    return np.array([i for i, w in enumerate(weeks) if a <= w <= b])


def factors(C, rows):
    """Seoul and per-dong factors for one window, using history C (already corrected before it)."""
    pre = [i for i in range(rows[0] - 1, -1, -1) if not anom_all[i] and i - LAG >= 0][:8]
    r_d = C[pre].sum(axis=0) / C[[i - LAG for i in pre]].sum(axis=0)
    r_s = C[pre].sum() / C[[i - LAG for i in pre]].sum()
    exp_d = C[rows - LAG] * r_d
    f_d = C[rows].sum(axis=0) / exp_d.sum(axis=0)
    f_s = C[rows].sum() / (C[rows - LAG].sum() * r_s)
    return f_s, f_d


def corrected(rule):
    C = W.copy()
    fac = {}
    if rule == "S1 midterm (2024 only, /1.52)":
        C[window_rows("2024-08~10")] /= 1.52
        return C, {"2024-08~10": 1.52}
    if rule == "S0 raw":
        return C, {}
    for k in P.ANOMALIES:
        rows = window_rows(k)
        f_s, f_d = factors(C, rows)
        if rule == "S2 Seoul factor":
            f = np.full(N, f_s)
        elif rule == "S3 dong factor":
            f = f_d
        else:
            f = np.exp(0.5 * np.log(f_d) + 0.5 * np.log(f_s))
        C[rows] = C[rows] / f
        fac[k] = {"seoul": round(float(f_s), 3), "dong_median": round(float(np.median(f_d)), 3),
                  "dong_p10_p90": [round(float(x), 3) for x in np.percentile(f_d, [10, 90])],
                  "dong_corr_with_size": round(P.spearman(f_d, W.mean(axis=0)), 3)}
    return C, fac


RULES = ["S0 raw", "S1 midterm (2024 only, /1.52)", "S2 Seoul factor", "S3 dong factor", "S4 dong factor, shrunk"]
widx = {str(w): i for i, w in enumerate(weeks)}
origins = [o for o in range(widx["2025-04-28"], n - H, 2) if not anom_all[o + 1:o + 1 + H].any()]

rows_out, summary = [], {"origins": [str(weeks[o]) for o in origins], "rules": {}}
for rule in RULES:
    C, fac = corrected(rule)
    acc = []
    for o in origins:
        Yf = W[o + 1:o + 1 + H]                                  # truth: raw, clean weeks
        ly = C[o + 1 - LAG:o + 1 + H - LAG]
        m13 = C[o - H + 1:o + 1].mean(axis=0)
        trend = C[o - H + 1:o + 1].sum() / C[o - H + 1 - LAG:o + 1 - LAG].sum()
        s_ly = P.share(C, o + 1 - LAG, o + H - LAG)
        g_mom = P.g_at(C, o)
        s_m = s_ly * np.exp(g_mom); s_m /= s_m.sum()
        B = {"same_week_last_year": ly, "same_week_last_year_x_trend": ly * trend,
             "avg(last_year, mean_13w)": (ly + m13) / 2, "share_momentum": (ly * trend).sum(axis=1, keepdims=True) * s_m}
        scale = np.abs(C[LAG:o + 1] - C[:o + 1 - LAG]).mean(axis=0)
        g_true = np.log(P.share(W, o + 1, o + H)) - np.log(s_ly)
        for b, F in B.items():
            e = np.abs(F - Yf)
            lvl = F.sum(axis=0)
            g_pred = np.log(lvl / lvl.sum()) - np.log(s_ly)
            gk, gt = g_pred[keep], g_true[keep]
            flat = np.allclose(gk, gk[0])
            r = {"rule": rule, "origin": str(weeks[o]), "baseline": b, "WAPE": e.sum() / Yf.sum(),
                 "MASE": float(np.mean(e.mean(axis=0) / scale)),
                 "share_WAPE": float(np.abs(F / F.sum(1, keepdims=True) - Yf / Yf.sum(1, keepdims=True)).sum() / H),
                 "change_spearman": float("nan") if flat else P.spearman(gk, gt),
                 "emerging_P@10": float("nan") if flat else len(P.topk(gk) & P.topk(gt)) / len(P.topk(gt))}
            rows_out.append(r); acc.append(r)
    summ = {}
    for b in dict.fromkeys(r["baseline"] for r in acc):
        rs = [r for r in acc if r["baseline"] == b]
        summ[b] = {k: (None if np.isnan([r[k] for r in rs]).all() else round(float(np.nanmean([r[k] for r in rs])), 4))
                   for k in ("WAPE", "MASE", "share_WAPE", "change_spearman", "emerging_P@10")}
    summary["rules"][rule] = {"factors": fac, "scores": summ}

with open(os.path.join(OUT, "scores_by_origin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader()
    for r in rows_out:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("origins:", summary["origins"])
for rule in RULES:
    print("\n==", rule, json.dumps(summary["rules"][rule]["factors"], ensure_ascii=False))
    for b, s in summary["rules"][rule]["scores"].items():
        print("   %-30s" % b, s)
