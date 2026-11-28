"""Ground-truth check: change in each dong's share of Seoul, vs the same 13 weeks one year earlier.

g[d, o] = log S_future[d] - log S_base[d]
  S_future = sum y[d, o+1..o+13] / sum Y[o+1..o+13]     (next 13 weeks after origin o)
  S_base   = sum y[d, o-51..o-39] / sum Y[o-51..o-39]   (the same 13 weeks, 52 weeks earlier)
y = weekly mean of daily arrivals, ages 20-39, purposes 4+5+7 (complete 7-day weeks only).

Checks
  A. How predictable is g? Momentum baseline (last 13 weeks' g) vs actual, rolling origins every 4 weeks.
     Note: the "same week last year" level forecast implies g = 0 for every dong, so it cannot rank change.
  B. Noise: standard error of g from weekly volatility; share of top-10% dongs with z >= 2.
  C. Label stability: overlap of the top-10% list between origins 4 and 13 weeks apart.
  D. The Aug-Oct 2024 anomaly: did dong shares move, or only the Seoul total?
  E. 20-39 vs all ages: agreement of g and of the top-10% lists.
  F. Other signals, calendar quarters vs one year earlier: card sales share (all ages and 20-39) and Naver search share.

Usage: python run_gt_share_change.py <codex_root> <heat_root> <out_dir>
"""
import csv, glob, io, json, os, struct, sys, zipfile
import numpy as np
import pandas as pd

ROOT, HEAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
H, LAG = 13, 52
res = {}

# ---------- weekly panel ----------
files = sorted(glob.glob(os.path.join(ROOT, "outputs", "nameplate_mobility_pilot", "age2030_daily_baseline", "daily_457_age2030_20????.csv")))
daily = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str}) for p in files], ignore_index=True)
daily["date"] = pd.to_datetime(daily.date.astype(str), format="%Y%m%d")
daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
wk = daily.groupby(["admin_dong_code", "week"]).agg(y=("age20_39", "mean"), a=("all_ages_457", "mean"), days=("date", "nunique")).reset_index()
wk = wk[wk.days.eq(7)]
n_weeks = wk.week.nunique()
full = wk.groupby("admin_dong_code").week.nunique()
wk = wk[wk.admin_dong_code.isin(full[full.eq(n_weeks)].index)]
Yp = wk.pivot(index="week", columns="admin_dong_code", values="y").sort_index()
Ap = wk.pivot(index="week", columns="admin_dong_code", values="a").sort_index()
weeks = Yp.index
res["panel"] = {"dongs": int(Yp.shape[1]), "weeks": int(len(weeks)), "first": str(weeks[0].date()), "last": str(weeks[-1].date())}

# size floor: 2023 bottom 25% of mean daily 20-39 arrivals (HEAT rule)
m23 = Yp[Yp.index.year == 2023].mean()
keep = m23.index[m23 >= m23.quantile(0.25)]
res["panel"]["dongs_above_floor"] = int(len(keep))


def share(P, i0, i1):  # rows i0..i1 inclusive
    blk = P.iloc[i0:i1 + 1]
    return blk.sum() / blk.sum().sum()


def g_at(P, i_end):  # 13-week window ending at row i_end vs 52 weeks earlier
    return np.log(share(P, i_end - H + 1, i_end)) - np.log(share(P, i_end - H + 1 - LAG, i_end - LAG))


def se_at(P, i_end):  # standard error of g from weekly log-share volatility in both windows
    S = np.log(P.div(P.sum(axis=1), axis=0))
    v1 = S.iloc[i_end - H + 1:i_end + 1].var()
    v0 = S.iloc[i_end - H + 1 - LAG:i_end - LAG + 1].var()
    return np.sqrt((v1 + v0) / H)


def top(s, q=0.10):
    return set(s.nlargest(int(round(len(s) * q))).index)


# ---------- A, B, C: rolling origins ----------
first_o = LAG + H - 1 + H          # momentum window needs its own base window
last_o = len(weeks) - 1 - H
origins = list(range(first_o, last_o + 1, 4))
rows, tops = [], {}
for o in origins:
    g_fut = g_at(Yp, o + H)[keep]
    g_mom = g_at(Yp, o)[keep]
    z = (g_fut / se_at(Yp, o + H)[keep])
    t_act, t_mom = top(g_fut), top(g_mom)
    tops[o] = t_act
    # any of the four windows (future, its base, momentum, its base) inside Aug-Oct 2024?
    win = [(o + 1, o + H), (o + 1 - LAG, o + H - LAG), (o - H + 1, o), (o - H + 1 - LAG, o - LAG)]
    a0, a1 = np.searchsorted(weeks, pd.Timestamp("2024-08-05")), np.searchsorted(weeks, pd.Timestamp("2024-10-21"))
    rows.append({"origin": str(weeks[o].date()),
                 "touches_anomaly": any(s <= a1 and e >= a0 for s, e in win),
                 "spearman_momentum": g_mom.rank().corr(g_fut.rank()),
                 "p_at_10_momentum": len(t_act & t_mom) / len(t_act),
                 "p_at_10_random": 0.10,
                 "sd_g": g_fut.std(), "median_abs_g": g_fut.abs().median(),
                 "top10_min_g": g_fut[list(t_act)].min(),
                 "top10_share_z_ge_2": float((z[list(t_act)] >= 2).mean())})
A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "rolling_origins.csv"), index=False, float_format="%.4f")
res["A_B_rolling"] = {"origins": len(A), "range": [A.origin.iloc[0], A.origin.iloc[-1]],
                      "spearman_momentum_median": float(A.spearman_momentum.median()),
                      "p_at_10_momentum_mean": float(A.p_at_10_momentum.mean()),
                      "sd_g_median": float(A.sd_g.median()),
                      "top10_min_g_median": float(A.top10_min_g.median()),
                      "top10_share_z_ge_2_mean": float(A.top10_share_z_ge_2.mean())}
Ac = A[~A.touches_anomaly]
res["A_B_rolling"]["clean_origins"] = Ac.origin.tolist()
res["A_B_rolling"]["clean_spearman_momentum_median"] = float(Ac.spearman_momentum.median()) if len(Ac) else None
res["A_B_rolling"]["clean_p_at_10_momentum_mean"] = float(Ac.p_at_10_momentum.mean()) if len(Ac) else None
ov = {}
for gap in (4, 8, 12, 52):
    vals = [len(tops[o] & tops[o + gap]) / len(tops[o]) for o in origins if o + gap in tops]
    if vals:
        ov[f"{gap}w"] = {"pairs": len(vals), "mean_overlap": float(np.mean(vals))}
res["C_top10_overlap"] = ov

# ---------- D: anomaly ----------
def idx(d): return int(np.searchsorted(weeks, pd.Timestamp(d)))
pre = share(Yp, idx("2024-06-03"), idx("2024-07-22"))[keep]
mid = share(Yp, idx("2024-08-05"), idx("2024-10-21"))[keep]
post = share(Yp, idx("2024-11-04"), idx("2024-12-23"))[keep]
pre23 = share(Yp, idx("2023-06-05"), idx("2023-07-24"))[keep]
mid23 = share(Yp, idx("2023-08-07"), idx("2023-10-23"))[keep]
tot = Yp.sum(axis=1)
res["D_anomaly"] = {
    "seoul_total_ratio_mid_vs_pre_2024": float(tot.iloc[idx("2024-08-05"):idx("2024-10-21") + 1].mean() / tot.iloc[idx("2024-06-03"):idx("2024-07-22") + 1].mean()),
    "seoul_total_ratio_mid_vs_pre_2023": float(tot.iloc[idx("2023-08-07"):idx("2023-10-23") + 1].mean() / tot.iloc[idx("2023-06-05"):idx("2023-07-24") + 1].mean()),
    "sd_log_share_shift_mid_vs_pre_2024": float(np.log(mid / pre).std()),
    "sd_log_share_shift_mid_vs_pre_2023": float(np.log(mid23 / pre23).std()),
    "spearman_shift_2024_vs_2023": float(np.log(mid / pre).rank().corr(np.log(mid23 / pre23).rank())),
    "sd_log_share_shift_post_vs_pre_2024": float(np.log(post / pre).std()),
}

# ---------- E: 20-39 vs all ages ----------
e = []
for o in origins:
    gy, ga = g_at(Yp, o + H)[keep], g_at(Ap, o + H)[keep]
    e.append({"spearman": gy.rank().corr(ga.rank()), "top10_overlap": len(top(gy) & top(ga)) / len(top(gy))})
E = pd.DataFrame(e)
res["E_age"] = {"spearman_median": float(E.spearman.median()), "top10_overlap_mean": float(E.top10_overlap.mean())}

# ---------- F: other signals by calendar quarter ----------
def qkey(ts): return f"{ts.year}{(ts.month - 1) // 3 + 1}"
qmob = Yp.groupby([qkey(t) for t in weeks]).sum()
qcnt = pd.Series([qkey(t) for t in weeks]).value_counts()
qmob = qmob[qmob.index.isin(qcnt[qcnt >= 12].index)]   # complete quarters only

REF = os.path.join(ROOT, "work", "Time-series-Project", "data", "reference")
z = zipfile.ZipFile(os.path.join(REF, "seoul_commercial_area_2023.zip"))
dbf = z.read([n for n in z.namelist() if n.endswith(".dbf")][0])
nrec, hlen, rlen = struct.unpack("<IHH", dbf[4:12])
fields, pos = [], 32
while dbf[pos] != 0x0D:
    fields.append((dbf[pos:pos + 11].split(b"\0")[0].decode(), dbf[pos + 16])); pos += 32
tr2dong = {}
for i in range(nrec):
    rec, off, vals = dbf[hlen + i * rlen: hlen + (i + 1) * rlen], 1, {}
    for name, ln in fields:
        vals[name] = rec[off:off + ln].strip(); off += ln
    tr2dong[vals["TRDAR_CD"].decode()] = vals["ADSTRD_CD"].decode()
card = {}
for yr in (2023, 2024, 2025):
    zz = zipfile.ZipFile(os.path.join(REF, f"seoul_commercial_sales_{yr}.zip"))
    for r in csv.DictReader(io.StringIO(zz.read(zz.namelist()[0]).decode("cp949"))):
        d = tr2dong.get(r["상권_코드"])
        if d:
            k = (r["기준_년분기_코드"], d)
            a = card.setdefault(k, [0.0, 0.0])
            a[0] += float(r["당월_매출_금액"]); a[1] += float(r["연령대_20_매출_금액"]) + float(r["연령대_30_매출_금액"])
C_all = pd.Series({k: v[0] for k, v in card.items()}).unstack()
C_y = pd.Series({k: v[1] for k, v in card.items()}).unstack()
sr = pd.read_csv(os.path.join(HEAT, "data", "processed", "search", "행정동_검색지수_월별.csv"), encoding="utf-8-sig", dtype={"dong_code8": str})
sr = sr[sr.tier.eq("전체")]
sr["q"] = sr.month.str[:4] + ((sr.month.str[5:7].astype(int) - 1) // 3 + 1).astype(str)
S_q = sr.groupby(["q", "dong_code8"])["index"].sum().unstack()


def gq(P, q):  # log share change of quarter q vs the same quarter one year earlier
    q0 = str(int(q[:4]) - 1) + q[4]
    if q not in P.index or q0 not in P.index:
        return None
    a, b = P.loc[q], P.loc[q0]
    ok = a.notna() & b.notna() & (a > 0) & (b > 0)
    a, b = a[ok], b[ok]
    return np.log(a / a.sum()) - np.log(b / b.sum())


f = []
for q in qmob.index:
    gm = gq(qmob, q)
    if gm is None:
        continue
    gm = gm[gm.index.isin(keep)]
    row = {"quarter": q}
    for name, P in (("card_all", C_all), ("card_20_39", C_y), ("search", S_q)):
        go = gq(P, q)
        if go is None:
            continue
        common = gm.index.intersection(go.index)
        row[f"{name}_n"] = len(common)
        row[f"{name}_spearman"] = gm[common].rank().corr(go[common].rank())
        tm, to = top(gm[common]), top(go[common])
        row[f"{name}_top10_overlap"] = len(tm & to) / len(tm)
    f.append(row)
F = pd.DataFrame(f)
F.to_csv(os.path.join(OUT, "other_signals_by_quarter.csv"), index=False, float_format="%.4f")
res["F_other_signals"] = F.drop(columns="quarter").mean(numeric_only=True).round(4).to_dict()
res["F_other_signals"]["quarters"] = F.quarter.tolist()

with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
