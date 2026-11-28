"""Does aligning Lunar New Year (설) and Chuseok (추석) improve the 'same week last year' reference?

The usual reference for week t is week t-52. The two lunar holidays move by up to ~3 weeks between years,
so t-52 can be a holiday week when t is not, and the reverse.
Aligned reference ref(t):
  - t touches a holiday block of year y: the week at the same offset from that holiday's main-day week in y-1
  - t touches no holiday but t-52 does: the nearest week to t-52 that touches no holiday (earlier side first)
  - otherwise t-52
Holiday days are the official public holidays (incl. substitute and temporary holidays) of 2023-2025.

Data: weekly mean, 20-39, purposes 4+5+7, to 2025-08-31 (P.DATA_END), history corrected with day-level S4. Truth weeks are
never in an anomaly window. 2026 is not used.
Checks
  A. per target week: WAPE and share error (L1 distance of Seoul shares) of C[t-52] vs C[ref(t)]
  B. 13-week windows: 'same week last year' baseline at origins whose 13 future weeks are clean
  C. target (g = change in Seoul share vs the same 13 weeks a year earlier): current vs aligned base window
Usage: python src/holiday/run_holiday_align.py <codex_root> <out_dir>
"""
import csv, json, os, sys
from datetime import date, timedelta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
H, LAG = P.H, P.LAG
d = date
HOLIDAYS = {  # name: (main day, all public holiday days of the block)
    "seollal_2023": (d(2023, 1, 22), [d(2023, 1, 21), d(2023, 1, 22), d(2023, 1, 23), d(2023, 1, 24)]),
    "seollal_2024": (d(2024, 2, 10), [d(2024, 2, 9), d(2024, 2, 10), d(2024, 2, 11), d(2024, 2, 12)]),
    "seollal_2025": (d(2025, 1, 29), [d(2025, 1, 27), d(2025, 1, 28), d(2025, 1, 29), d(2025, 1, 30)]),
    "chuseok_2023": (d(2023, 9, 29), [d(2023, 9, 28), d(2023, 9, 29), d(2023, 9, 30), d(2023, 10, 2), d(2023, 10, 3)]),
    "chuseok_2024": (d(2024, 9, 17), [d(2024, 9, 16), d(2024, 9, 17), d(2024, 9, 18)]),
    "chuseok_2025": (d(2025, 10, 6), [d(2025, 10, 3), d(2025, 10, 5), d(2025, 10, 6), d(2025, 10, 7), d(2025, 10, 8), d(2025, 10, 9)]),
}

weeks, dongs, W, C = P.load_weekly_extended(ROOT)   # data to 2025-08 (P.DATA_END), C = day-level S4
n, N = W.shape
anom = P.anomaly_mask(weeks)
keep = P.size_floor(weeks, W)
widx = {w: i for i, w in enumerate(weeks)}
monday = lambda x: x - timedelta(days=x.weekday())

touch = {}  # week index -> (holiday name, offset in weeks from the main-day week)
for name, (main, days) in HOLIDAYS.items():
    for x in days:
        w = monday(x)
        if w in widx:
            touch[widx[w]] = (name, (w - monday(main)).days // 7)


def ref(t):
    if t in touch:
        name, off = touch[t]
        kind, y = name.split("_")
        prev = HOLIDAYS.get(f"{kind}_{int(y) - 1}")
        if prev:
            w = monday(prev[0]) + timedelta(weeks=off)
            if w in widx:
                return widx[w], "holiday"
        return t - LAG, "holiday_unmapped"
    if t - LAG in touch:
        for k in (1, 2, 3):
            for j in (t - LAG - k, t - LAG + k):
                if 0 <= j < n and j not in touch:
                    return j, "shifted"
    return t - LAG, "same"


R = {t: ref(t) for t in range(LAG, n)}

# A. per target week
rows, groups = [], {}
for t in range(LAG, n):
    if anom[t]:
        continue
    r, kind = R[t]
    Y = W[t]
    out = {"week": str(weeks[t]), "kind": kind, "ref_week": str(weeks[r]), "t_minus_52": str(weeks[t - LAG])}
    for m, F in (("t-52", C[t - LAG]), ("aligned", C[r])):
        out[f"WAPE_{m}"] = float(np.abs(F - Y).sum() / Y.sum())
        out[f"share_err_{m}"] = float(np.abs(F / F.sum() - Y / Y.sum()).sum())
        out[f"share_err_keep_{m}"] = float(np.abs(F[keep] / F[keep].sum() - Y[keep] / Y[keep].sum()).sum())
    rows.append(out)
    groups.setdefault(kind, []).append(out)
with open(os.path.join(OUT, "by_week.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
summ_A = {k: {"weeks": len(v), **{m: round(float(np.mean([r[m] for r in v])), 4) for m in
                                  ("WAPE_t-52", "WAPE_aligned", "share_err_t-52", "share_err_aligned")}}
          for k, v in groups.items()}

# B. 13-week windows, 'same week last year' baseline
origins = [o for o in range(LAG - 1, n - H) if not anom[o + 1:o + 1 + H].any()]
B = []
for o in origins:
    Yf = W[o + 1:o + 1 + H]
    fut = range(o + 1, o + 1 + H)
    res = {"origin": str(weeks[o]), "holiday_related": any(R[t][1] != "same" for t in fut)}
    for m, idx in (("t-52", [t - LAG for t in fut]), ("aligned", [R[t][0] for t in fut])):
        F = C[idx]
        res[f"WAPE_{m}"] = float(np.abs(F - Yf).sum() / Yf.sum())
        res[f"share_WAPE_{m}"] = float(np.abs(F / F.sum(1, keepdims=True) - Yf / Yf.sum(1, keepdims=True)).sum() / H)
    B.append(res)
summ_B = {}
for label, sel in (("all clean origins", B), ("windows with a holiday-related week", [r for r in B if r["holiday_related"]])):
    summ_B[label] = {"origins": len(sel), **{m: round(float(np.mean([r[m] for r in sel])), 4) for m in
                                             ("WAPE_t-52", "WAPE_aligned", "share_WAPE_t-52", "share_WAPE_aligned")}}

# C. target sensitivity: base window t-52 vs aligned, for origins whose 13 future weeks are clean
C_rows = []
for o in origins:
    fut = list(range(o + 1, o + 1 + H))
    if all(R[t][1] == "same" for t in fut):
        continue
    s_fut = W[fut].sum(0)
    g_now = np.log(s_fut / s_fut.sum()) - np.log(P.share(C, fut[0] - LAG, fut[-1] - LAG))
    b = C[[R[t][0] for t in fut]].sum(0)
    g_al = np.log(s_fut / s_fut.sum()) - np.log(b / b.sum())
    a, c = g_now[keep], g_al[keep]
    C_rows.append({"origin": str(weeks[o]), "spearman": P.spearman(a, c),
                   "top10_overlap": len(P.topk(a) & P.topk(c)) / len(P.topk(a)),
                   "mean_abs_diff": float(np.abs(a - c).mean()), "mean_abs_g": float(np.abs(a).mean())})
summ_C = {"origins_with_holiday_in_window": len(C_rows),
          **{k: round(float(np.mean([r[k] for r in C_rows])), 4) for k in ("spearman", "top10_overlap", "mean_abs_diff", "mean_abs_g")},
          "top10_overlap_min": round(float(min(r["top10_overlap"] for r in C_rows)), 4)} if C_rows else {}

summary = {"reference_map": {str(weeks[t]): [str(weeks[r]), k] for t, (r, k) in R.items() if k != "same"},
           "A_per_week": summ_A, "B_13w_windows": summ_B, "C_target": summ_C, "C_by_origin": C_rows}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "C_by_origin"}, ensure_ascii=False, indent=1))
