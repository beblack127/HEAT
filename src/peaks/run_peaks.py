"""One-off peaks (spikes) in daily arrivals: find them, check that they look like real events, see whether
they repeat a year later, and test whether removing them from the history improves simple forecasts.

Spike rule (per dong, per day, ages 20-39, purposes 4+5+7):
  baseline = median of the same weekday in the 4 weeks before and 4 weeks after (8 values, the day excluded)
  scale    = 1.4826 * median absolute deviation of those 8 values (floor: 5% of the baseline)
  spike    = y >= 1.5 * baseline and (y - baseline) / scale >= 5, on at most 3 consecutive days
  citywide = more than 15% of dongs spike on the same day (holidays, data issues); kept but flagged
Usage: python src/peaks/run_peaks.py <codex_root> <out_dir>
"""
import csv, json, os, sys
from collections import Counter, defaultdict
from datetime import timedelta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dates, dongs, Y, _ = P.load_daily(ROOT)
nm = P.names()
T, N = Y.shape
day_i = {d: i for i, d in enumerate(dates)}

# ---------- detection ----------
base = np.full_like(Y, np.nan); scale = np.full_like(Y, np.nan)
for t in range(T):
    nb = [day_i.get(dates[t] + timedelta(weeks=k)) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    nb = [i for i in nb if i is not None]
    if len(nb) < 6:
        continue
    v = Y[nb]
    med = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - med), axis=0)
    base[t], scale[t] = med, np.maximum(mad, 0.05 * med)
with np.errstate(divide="ignore", invalid="ignore"):
    z = (Y - base) / scale
    ratio = Y / base
raw_flag = (ratio >= 1.5) & (z >= 5) & np.isfinite(z)
# keep runs of at most 3 consecutive days
flag = np.zeros_like(raw_flag)
for j in range(N):
    t = 0
    while t < T:
        if raw_flag[t, j]:
            s = t
            while t + 1 < T and raw_flag[t + 1, j]:
                t += 1
            if t - s + 1 <= 3:
                flag[s:t + 1, j] = True
        t += 1
share_day = flag.mean(axis=1)
citywide_days = {dates[t] for t in range(T) if share_day[t] > 0.15}

events = []
for t, j in zip(*np.nonzero(flag)):
    events.append({"date": str(dates[t]), "weekday": "월화수목금토일"[dates[t].weekday()], "dong": dongs[j], "name": nm.get(dongs[j], dongs[j]),
                   "arrivals": round(float(Y[t, j])), "baseline": round(float(base[t, j])), "ratio": round(float(ratio[t, j]), 2),
                   "z": round(float(z[t, j]), 1), "extra_people": round(float(Y[t, j] - base[t, j])), "citywide_day": dates[t] in citywide_days})
events.sort(key=lambda e: -e["extra_people"])
with open(os.path.join(OUT, "spike_events.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)

local = [e for e in events if not e["citywide_day"]]
by_dong = Counter(e["name"] for e in local)

# ---------- does a spike repeat about a year later? ----------
# spike week in the first year -> any local spike in the same dong within +-1 week, 52 weeks later
spike_set = {(e["dong"], e["date"]) for e in local}
from datetime import date as _d
def week_spikes(dong, center, half=10):
    c = _d.fromisoformat(center)
    return any((dong, str(c + timedelta(days=k))) in spike_set for k in range(-half, half + 1))
first = [e for e in local if _d.fromisoformat(e["date"]) + timedelta(weeks=52) <= dates[-1]]
repeat = [week_spikes(e["dong"], str(_d.fromisoformat(e["date"]) + timedelta(weeks=52))) for e in first]
# chance level: same check at a random offset of 26 weeks
chance = [week_spikes(e["dong"], str(_d.fromisoformat(e["date"]) + timedelta(weeks=26))) for e in first if _d.fromisoformat(e["date"]) + timedelta(weeks=26) <= dates[-1]]

# ---------- does removing spikes from the history help forecasts? ----------
Yc = np.where(flag, base, Y)                          # spikes replaced by their baseline
weeks, W = P.weekly(dates, Y)
_, Wc = P.weekly(dates, Yc)
H, LAG = P.H, P.LAG
keep = P.size_floor(weeks, W)
anom = P.anomaly_rows(weeks)
res = defaultdict(list)
for o in range(LAG + 2 * H - 1, len(weeks) - H, 4):
    Yf = W[o + 1:o + 1 + H]                           # truth stays raw
    for hist_name, Hh in (("raw_history", W), ("spikes_removed", Wc)):
        ly = Hh[o + 1 - LAG:o + 1 + H - LAG]; m13 = Hh[o - H + 1:o + 1].mean(axis=0)
        for b, F in (("same_week_last_year", ly), ("avg(last_year, mean_13w)", (ly + m13) / 2), ("last_week", np.repeat(Hh[o:o + 1], H, axis=0))):
            e = np.abs(F - Yf)
            lvl = F.sum(axis=0)
            g_true = np.log(P.share(W, o + 1, o + H)) - np.log(P.share(W, o + 1 - LAG, o + H - LAG))
            g_pred = np.log(lvl / lvl.sum()) - np.log(P.share(W, o + 1 - LAG, o + H - LAG))
            gk = g_pred[keep]
            cs = float("nan") if np.allclose(gk, gk[0]) else P.spearman(gk, g_true[keep])
            res[(hist_name, b)].append((e.sum() / Yf.sum(), cs))
eff = {f"{h} | {b}": {"WAPE": round(float(np.mean([v[0] for v in vals])), 4),
                      "change_spearman": None if np.isnan([v[1] for v in vals]).all() else round(float(np.nanmean([v[1] for v in vals])), 4)}
       for (h, b), vals in res.items()}

summary = {
    "days": T, "dongs": N, "spike_days_total": int(flag.sum()), "citywide_days": sorted(str(d) for d in citywide_days),
    "local_spikes": len(local), "dongs_with_local_spike": len(by_dong),
    "spike_share_of_dong_days": round(float(flag.mean()), 5),
    "top_dongs_by_local_spikes": by_dong.most_common(20),
    "repeat_after_52w": {"spikes_checked": len(first), "repeat_rate": round(float(np.mean(repeat)), 3) if first else None,
                         "chance_rate_26w": round(float(np.mean(chance)), 3) if chance else None},
    "effect_on_forecasts_13_origins": eff,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("\nTop 30 local spikes:")
for e in local[:30]:
    print(e["date"], e["weekday"], e["name"], e["arrivals"], "vs", e["baseline"], "x", e["ratio"])
