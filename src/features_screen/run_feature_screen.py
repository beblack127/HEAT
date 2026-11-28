"""Feature catalog and signal screening (no forecasting model).

For every origin week o and dong, build features that are known at o, save them in one table keyed by
(dong_code8, origin_week), and check how each one relates to the change that happens next:
  g_future = log share of weeks o+1..o+13 - log share of the same 13 weeks one year earlier
Screening uses rank correlation across dongs at each origin, and the same after removing what momentum
(the last 13 weeks' change) already explains, so a feature only scores if it adds something new.

Publication lags assumed (stated in the report): living population and subway 1 month, Naver search
4 months (midterm rule), card sales 1 quarter.
Usage: python src/features_screen/run_feature_screen.py <codex_root> <out_dir>
"""
import csv, io, json, os, struct, sys, zipfile
from collections import defaultdict
from datetime import date, timedelta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gt_refine"))
import panel as P

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
TS = os.path.join(ROOT, "work", "Time-series-Project", "data")
H, LAG = P.H, P.LAG
dates, dongs, Yd, Ad = P.load_daily(ROOT)
ci = {c: j for j, c in enumerate(dongs)}
weeks, W = P.weekly(dates, Yd)
_, Wall = P.weekly(dates, Ad)
_, Wwe = P.weekly(dates, Yd, days_of_week=(5, 6))
_, Wwd = P.weekly(dates, Yd, days_of_week=(0, 1, 2, 3, 4))
keep = P.size_floor(weeks, W)
anom = P.anomaly_rows(weeks)
n, N = len(weeks), len(dongs)

# spikes (same rule as src/peaks), as a weekly count per dong
day_i = {d: i for i, d in enumerate(dates)}
spike = np.zeros_like(Yd, dtype=bool)
for t in range(len(dates)):
    nb = [day_i.get(dates[t] + timedelta(weeks=k)) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    nb = [i for i in nb if i is not None]
    if len(nb) < 6:
        continue
    v = Yd[nb]; med = np.median(v, axis=0); sc = np.maximum(1.4826 * np.median(np.abs(v - med), axis=0), 0.05 * med)
    spike[t] = (Yd[t] >= 1.5 * med) & ((Yd[t] - med) / sc >= 5)
city = spike.mean(axis=1) > 0.15
spike[city] = False
_, Wsp = P.weekly(dates, spike.astype(float))
Wsp = Wsp * 7


# ---------- monthly / quarterly external data ----------
def month_add(ym, k):
    y, m = int(ym[:4]), int(ym[5:7]) + k
    while m <= 0: y, m = y - 1, m + 12
    while m > 12: y, m = y + 1, m - 12
    return f"{y}-{m:02d}"


def load_monthly(path_glob_rows, key_col, val_col):
    out = defaultdict(lambda: np.full(N, np.nan))
    for r in path_glob_rows:
        j = ci.get(r[key_col])
        if j is not None and r[val_col] not in ("", None):
            out[r["month"]][j] = float(r[val_col])
    return out


def rows_of(paths):
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)


import glob
liv_paths = sorted(glob.glob(os.path.join(TS, "context_monthly", "living_*.csv")))
LIV = {c: load_monthly(rows_of(liv_paths), "admin_dong_code", c) for c in ("living_all_hour_mean", "living_night_mean", "living_weekend_day_mean")}
SR = defaultdict(lambda: np.full(N, np.nan))
for r in rows_of([os.path.join(P.HEAT, "data", "processed", "search", "행정동_검색지수_월별.csv")]):
    if r["tier"] == "전체" and r["dong_code8"] in ci and r["seoul_share"]:
        SR[r["month"]][ci[r["dong_code8"]]] = float(r["seoul_share"])
# subway: station -> nearest dong (link table), monthly exits per day
best = {}
for r in rows_of([os.path.join(P.HEAT, "data", "processed", "nametags", "05_행정동_역_연결표.csv")]):
    k = (float(r["dist_to_dong_m"]), -float(r["share_500"] or 0))
    if r["station"] not in best or k < best[r["station"]][0]:
        best[r["station"]] = (k, r["dong_code8"])
SUB = defaultdict(lambda: np.zeros(N))
for r in rows_of([os.path.join(TS, "context_monthly", "subway_station_monthly_2023_2025.csv")]):
    d = best.get(r["station_name"])
    if d and d[1] in ci and float(r["days"]) > 0:
        SUB[r["month"]][ci[d[1]]] += float(r["exits"]) / float(r["days"])
# card: 20-39 spending in hot-place business types (same list as reports/gt_refine/card_hotplace), by quarter
HOT = ["커피-음료", "제과점", "양식음식점", "일식음식점", "호프-간이주점", "일반의류", "신발", "가방", "화장품", "시계및귀금속", "노래방"]
z = zipfile.ZipFile(os.path.join(TS, "reference", "seoul_commercial_area_2023.zip"))
dbf = z.read([x for x in z.namelist() if x.endswith(".dbf")][0])
nrec, hlen, rlen = struct.unpack("<IHH", dbf[4:12]); fl, p = [], 32
while dbf[p] != 0x0D:
    fl.append((dbf[p:p + 11].split(b"\0")[0].decode(), dbf[p + 16])); p += 32
tr2dong = {}
for i in range(nrec):
    rec, o_, v = dbf[hlen + i * rlen:hlen + (i + 1) * rlen], 1, {}
    for nm_, ln in fl:
        v[nm_] = rec[o_:o_ + ln].strip(); o_ += ln
    tr2dong[v["TRDAR_CD"].decode()] = v["ADSTRD_CD"].decode()
CARD = defaultdict(lambda: np.zeros(N))
for yr in (2023, 2024, 2025):
    zz = zipfile.ZipFile(os.path.join(TS, "reference", f"seoul_commercial_sales_{yr}.zip"))
    for r in csv.DictReader(io.StringIO(zz.read(zz.namelist()[0]).decode("cp949"))):
        d = tr2dong.get(r["상권_코드"])
        if d in ci and r["서비스_업종_코드_명"] in HOT:
            CARD[r["기준_년분기_코드"]][ci[d]] += float(r["연령대_20_매출_금액"]) + float(r["연령대_30_매출_금액"])


def yoy_share(table, k_now, k_prev):
    if k_now not in table or k_prev not in table:
        return np.full(N, np.nan)
    a, b = table[k_now].astype(float).copy(), table[k_prev].astype(float).copy()
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out = np.full(N, np.nan)
    out[ok] = np.log(a[ok] / a[ok].sum()) - np.log(b[ok] / b[ok].sum())
    return out


def level_share(table, k):
    if k not in table:
        return np.full(N, np.nan)
    a = table[k].astype(float)
    return a / np.nansum(a)


def slope(Y):  # OLS slope over rows, per column
    t = np.arange(len(Y)) - (len(Y) - 1) / 2
    return (t[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (t ** 2).sum()


def features(o):
    w = weeks[o]
    ls = np.log(W / W.sum(axis=1, keepdims=True))
    m_now = f"{w.year}-{w.month:02d}"
    m1 = month_add(m_now, -1)                    # last month assumed published at o
    m4 = month_add(m_now, -4)                    # search: midterm rule
    q_last = (date(w.year, w.month, 1) - timedelta(days=95))  # quarter fully published
    qk = f"{q_last.year}{(q_last.month - 1) // 3 + 1}"
    qk0 = f"{int(qk[:4]) - 1}{qk[4]}"
    f = {
        "mob_g_13w (momentum)": P.g_at(W, o),
        "mob_g_4w": np.log(P.share(W, o - 3, o)) - np.log(P.share(W, o - 3 - LAG, o - LAG)),
        "mob_g_26w": np.log(P.share(W, o - 25, o)) - np.log(P.share(W, o - 25 - LAG, o - LAG)),
        "mob_share_slope_13w": slope(ls[o - 12:o + 1]),
        "mob_share_slope_26w": slope(ls[o - 25:o + 1]),
        "mob_volatility_26w": ls[o - 25:o + 1].std(axis=0),
        "mob_log_size_13w": np.log(W[o - 12:o + 1].mean(axis=0)),
        "mob_weekend_ratio_13w": Wwe[o - 12:o + 1].mean(axis=0) / Wwd[o - 12:o + 1].mean(axis=0),
        "mob_weekend_ratio_change": np.log(Wwe[o - 12:o + 1].mean(axis=0) / Wwd[o - 12:o + 1].mean(axis=0))
                                    - np.log(Wwe[o - 12 - LAG:o + 1 - LAG].mean(axis=0) / Wwd[o - 12 - LAG:o + 1 - LAG].mean(axis=0)),
        "mob_young_share_13w": W[o - 12:o + 1].sum(axis=0) / Wall[o - 12:o + 1].sum(axis=0),
        "mob_young_share_change": np.log(W[o - 12:o + 1].sum(axis=0) / Wall[o - 12:o + 1].sum(axis=0))
                                  - np.log(W[o - 12 - LAG:o + 1 - LAG].sum(axis=0) / Wall[o - 12 - LAG:o + 1 - LAG].sum(axis=0)),
        "spikes_last_52w": Wsp[o - 51:o + 1].sum(axis=0),
        "spikes_same_window_last_year": Wsp[o + 1 - LAG:o + 1 + H - LAG].sum(axis=0),
        "living_g_yoy (m-1)": yoy_share(LIV["living_all_hour_mean"], m1, month_add(m1, -12)),
        "living_weekend_day_g_yoy (m-1)": yoy_share(LIV["living_weekend_day_mean"], m1, month_add(m1, -12)),
        "living_night_to_all (m-1)": LIV["living_night_mean"][m1] / LIV["living_all_hour_mean"][m1] if m1 in LIV["living_night_mean"] else np.full(N, np.nan),
        "subway_g_yoy (m-1)": yoy_share(SUB, m1, month_add(m1, -12)),
        "search_g_yoy (m-4)": yoy_share(SR, m4, month_add(m4, -12)),
        "search_share (m-4)": level_share(SR, m4),
        "card_young_hot_g_yoy (q-1)": yoy_share(CARD, qk, qk0),
    }
    return f


first_o, last_o = LAG + 2 * H - 1, n - 1 - H
origins = list(range(first_o, last_o + 1, 4))
table_rows, scr = [], defaultdict(lambda: {"raw": [], "beyond_momentum": [], "clean_raw": [], "clean_beyond": [], "coverage": []})
for o in origins:
    f = features(o)
    g_fut = np.log(P.share(W, o + 1, o + H)) - np.log(P.share(W, o + 1 - LAG, o + H - LAG))
    mom = f["mob_g_13w (momentum)"]
    k = keep & np.isfinite(mom)
    A_ = np.c_[np.ones(k.sum()), mom[k]]
    beta = np.linalg.lstsq(A_, g_fut[k], rcond=None)[0]
    resid = np.full(N, np.nan); resid[k] = g_fut[k] - A_ @ beta
    win = [(o + 1, o + H), (o + 1 - LAG, o + H - LAG), (o - H + 1, o), (o - H + 1 - LAG, o - LAG)]
    clean = not any(anom[a:b + 1].any() for a, b in win)
    for name, v in f.items():
        m = keep & np.isfinite(v)
        if m.sum() < 50:
            continue
        r1 = P.spearman(v[m], g_fut[m]); r2 = P.spearman(v[m], resid[m]) if name != "mob_g_13w (momentum)" else float("nan")
        s = scr[name]; s["raw"].append(r1); s["beyond_momentum"].append(r2); s["coverage"].append(int(m.sum()))
        if clean:
            s["clean_raw"].append(r1); s["clean_beyond"].append(r2)
    for j, c in enumerate(dongs):
        table_rows.append([c, str(weeks[o]), int(clean), round(float(g_fut[j]), 5)] + [("" if not np.isfinite(f[x][j]) else round(float(f[x][j]), 6)) for x in f])

fnames = list(features(origins[0]))
with open(os.path.join(OUT, "feature_table.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["dong_code8", "origin_week", "clean_origin", "g_future"] + fnames); w.writerows(table_rows)


def mean(x):
    x = [v for v in x if np.isfinite(v)]
    return round(float(np.mean(x)), 3) if x else None


res = {nm_: {"spearman_with_future_change": mean(s["raw"]), "spearman_beyond_momentum": mean(s["beyond_momentum"]),
             "clean_spearman": mean(s["clean_raw"]), "clean_beyond_momentum": mean(s["clean_beyond"]),
             "origins": len(s["raw"]), "dongs_covered_median": int(np.median(s["coverage"]))} for nm_, s in scr.items()}
res = dict(sorted(res.items(), key=lambda kv: -abs(kv[1]["spearman_beyond_momentum"] or 0)))
json.dump({"origins": [str(weeks[o]) for o in origins], "dongs_above_floor": int(keep.sum()), "features": res},
          open(os.path.join(OUT, "screening.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("%-36s %8s %8s %8s %8s %4s %5s" % ("feature", "raw", "beyondM", "c_raw", "c_beyM", "n_o", "dongs"))
for k_, v in res.items():
    print("%-36s %8s %8s %8s %8s %4d %5d" % (k_, v["spearman_with_future_change"], v["spearman_beyond_momentum"], v["clean_spearman"], v["clean_beyond_momentum"], v["origins"], v["dongs_covered_median"]))
