"""Which metric should replace or sit next to WAPE? Score simple baselines with many metrics and see
which metrics separate "good at level" from "good at finding change".

No models are trained: the forecasts are simple baselines, so the comparison is about the metrics.
Target: weekly mean daily arrivals, ages 20-39, purposes 4+5+7 (see src/gt_refine/panel.py).
Origins: every 4 weeks, 13-week windows, same as reports/gt_refine/share_change.

Usage: python src/eval_metrics/run_metric_comparison.py <codex_root> <out_dir>
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
H, LAG = P.H, P.LAG
dates, dongs, Yd, _ = P.load_daily(ROOT)
weeks, W = P.weekly(dates, Yd)
n = len(weeks)
keep = P.size_floor(weeks, W)
anom = P.anomaly_rows(weeks)


def baselines(o):
    """Level forecasts for weeks o+1..o+13, shape (13, dongs)."""
    hist = W[:o + 1]
    last_year = W[o + 1 - LAG:o + 1 + H - LAG]
    seoul_ratio = W[o - H + 1:o + 1].sum() / W[o - H + 1 - LAG:o + 1 - LAG].sum()
    s_ly = share(W, o + 1 - LAG, o + H - LAG)
    g_mom = P.g_at(W, o)
    s_mom = s_ly * np.exp(g_mom); s_mom /= s_mom.sum()
    total = (last_year * seoul_ratio).sum(axis=1, keepdims=True)
    return {
        "last_week": np.repeat(hist[-1:], H, axis=0),
        "mean_last_13w": np.repeat(hist[-H:].mean(axis=0, keepdims=True), H, axis=0),
        "same_week_last_year": last_year,
        "same_week_last_year_x_seoul_trend": last_year * seoul_ratio,
        "share_momentum": total * s_mom,
        "avg(last_year, mean_13w)": (last_year + hist[-H:].mean(axis=0)) / 2,
    }


def share(X, i0, i1):
    return P.share(X, i0, i1)


def metrics(o, F):
    Yf = W[o + 1:o + 1 + H]
    e = np.abs(F - Yf)
    # MASE scale: in-sample mean |y_t - y_{t-52}| per dong, training weeks only
    scale = np.abs(W[LAG:o + 1] - W[:o + 1 - LAG]).mean(axis=0)
    sh_f, sh_y = F / F.sum(axis=1, keepdims=True), Yf / Yf.sum(axis=1, keepdims=True)
    lvl_f, lvl_y = F.sum(axis=0), Yf.sum(axis=0)
    g_true = np.log(share(W, o + 1, o + H)) - np.log(share(W, o + 1 - LAG, o + H - LAG))
    g_pred = np.log(lvl_f / lvl_f.sum()) - np.log(share(W, o + 1 - LAG, o + H - LAG))
    gk, gtk = g_pred[keep], g_true[keep]
    flat = np.allclose(gk, gk[0], atol=1e-9)
    per_dong_wape = e.sum(axis=0) / Yf.sum(axis=0)
    return {
        "WAPE": e.sum() / Yf.sum(),
        "MASE": float(np.mean(e.mean(axis=0) / scale)),
        "median_dong_WAPE": float(np.median(per_dong_wape)),
        "share_WAPE": np.abs(sh_f - sh_y).sum() / sh_y.sum(),
        "level_rank_spearman": P.spearman(lvl_f, lvl_y),
        "hot_P@10": len(P.topk(lvl_f) & P.topk(lvl_y)) / len(P.topk(lvl_y)),
        "change_spearman": float("nan") if flat else P.spearman(gk, gtk),
        "emerging_P@10": float("nan") if flat else len(P.topk(gk) & P.topk(gtk)) / len(P.topk(gtk)),
    }


first_o, last_o = LAG + 2 * H - 1, n - 1 - H
origins = list(range(first_o, last_o + 1, 4))
rows = []
for o in origins:
    # any of the windows used (future, its base, recent 13 weeks, its base) inside Aug-Oct 2024?
    win = [(o + 1, o + H), (o + 1 - LAG, o + H - LAG), (o - H + 1, o), (o - H + 1 - LAG, o - LAG)]
    touches = any(anom[a:b + 1].any() for a, b in win)
    for name, F in baselines(o).items():
        rows.append({"origin": str(weeks[o]), "touches_anomaly": bool(touches), "baseline": name, **metrics(o, F)})
keys = [k for k in rows[0] if k not in ("origin", "touches_anomaly", "baseline")]
with open(os.path.join(OUT, "metrics_by_origin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

summary = {}
for subset, cond in (("all_origins", lambda r: True), ("clean_origins", lambda r: not r["touches_anomaly"])):
    summary[subset] = {}
    for b in dict.fromkeys(r["baseline"] for r in rows):
        rs = [r for r in rows if r["baseline"] == b and cond(r)]
        summary[subset][b] = {k: (None if np.all(np.isnan([r[k] for r in rs])) else round(float(np.nanmean([r[k] for r in rs])), 4)) for k in keys}
    summary[subset]["n_origins"] = len({r["origin"] for r in rows if cond(r)})
# do metrics agree on the ordering of baselines?
b_names = list(dict.fromkeys(r["baseline"] for r in rows))
mat = np.array([[summary["all_origins"][b][k] if summary["all_origins"][b][k] is not None else np.nan for k in keys] for b in b_names])
sign = {k: (1 if k in ("WAPE", "MASE", "median_dong_WAPE", "share_WAPE") else -1) for k in keys}  # 1 = lower is better
order = {k: [b_names[i] for i in np.argsort(sign[k] * np.nan_to_num(mat[:, j], nan=np.inf * sign[k]))] for j, k in enumerate(keys)}
summary["best_first_by_metric"] = order
summary["dongs"] = len(dongs); summary["dongs_above_floor"] = int(keep.sum())
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
