"""Shared weekly panel for 재민's checks (numpy + csv only; pandas is blocked on 재민's PC).

Daily source: <codex_root>/outputs/nameplate_mobility_pilot/age2030_daily_baseline/daily_457_age2030_YYYYMM.csv
  columns: date, admin_dong_code, age20, age30, age20_39, all_ages_457 (daily arrivals, purposes 4+5+7)
Weeks run Monday-Sunday and are keyed by the Monday. Only weeks with all 7 days are kept, and only
dongs present on every day.
"""
import csv, glob, os
from datetime import date, timedelta
import numpy as np

HEAT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = 13      # forecast window (weeks)
LAG = 52    # one year

# Reference lists of known hot places (dong codes)
# Careet (대학내일) "Z세대가 자주 찾는 서울 동네" (https://www.careet.net/902), mapped to dongs by 재민's agent.
# 성수 was left out of the article as "already famous"; it is added separately and marked.
CAREET = {"11440660": "홍대(서교동)", "11410615": "연희동", "11110515": "서촌(청운효자동)", "11110530": "서촌(사직동)",
          "11110540": "북촌·안국(삼청동)", "11110600": "북촌·안국(가회동)", "11140550": "명동", "11140540": "회현동",
          "11140615": "신당동", "11170625": "삼각지(한강로동)", "11170685": "한남동", "11680545": "압구정로데오(압구정동)",
          "11680510": "압구정로데오(신사동)", "11710710": "잠실(잠실6동)"}
SEONGSU = {"11200670": "성수2가1동", "11200690": "성수2가3동"}


def load_daily(root):
    files = sorted(glob.glob(os.path.join(root, "outputs", "nameplate_mobility_pilot", "age2030_daily_baseline", "daily_457_age2030_20????.csv")))
    young, allg = {}, {}
    for p in files:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["date"], r["admin_dong_code"])
                young[k] = float(r["age20_39"]); allg[k] = float(r["all_ages_457"])
    days = sorted({d for d, _ in young}); dongs = sorted({c for _, c in young})
    di = {d: i for i, d in enumerate(days)}; ci = {c: j for j, c in enumerate(dongs)}
    Y = np.full((len(days), len(dongs)), np.nan); A = np.full_like(Y, np.nan)
    for (d, c), v in young.items():
        Y[di[d], ci[c]] = v; A[di[d], ci[c]] = allg[(d, c)]
    full = ~np.isnan(Y).any(axis=0)
    dates = [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in days]
    return dates, [c for c, k in zip(dongs, full) if k], Y[:, full], A[:, full]


def weekly(dates, X, days_of_week=range(7)):
    """Weekly mean over the chosen weekdays (0=Mon .. 6=Sun), complete weeks only."""
    dows = set(days_of_week)
    start = dates[0] + timedelta(days=(7 - dates[0].weekday()) % 7)
    idx = {d: i for i, d in enumerate(dates)}
    weeks, rows = [], []
    w = start
    while w + timedelta(days=6) <= dates[-1]:
        ii = [idx.get(w + timedelta(days=k)) for k in range(7)]
        if None not in ii:
            sel = [ii[k] for k in range(7) if k in dows]
            weeks.append(w); rows.append(X[sel].mean(axis=0))
        w += timedelta(days=7)
    return weeks, np.vstack(rows)


def names():
    out = {}
    with open(os.path.join(HEAT, "data", "processed", "nametags", "01_행정동_목록.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["dong_code8"]] = r["gu"] + " " + r["dong_name"]
    return out


def ref_121(categories=("발달상권", "관광특구")):
    """Dongs holding Seoul's official major places (from reports/gt_refine/card_hotplace/place_to_dong.csv)."""
    out = set()
    with open(os.path.join(HEAT, "reports", "gt_refine", "card_hotplace", "place_to_dong.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["category"] in categories:
                out.add(r["dong"])
    return out


def share(W, i0, i1):
    """Share of Seoul over rows i0..i1 inclusive (ratio of sums)."""
    s = W[i0:i1 + 1].sum(axis=0)
    return s / s.sum()


def g_at(W, i_end):
    """log share of the 13 weeks ending at i_end minus that of the same 13 weeks one year earlier."""
    return np.log(share(W, i_end - H + 1, i_end)) - np.log(share(W, i_end - H + 1 - LAG, i_end - LAG))


def rank(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v))
    # average ties
    vs = v[o]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and vs[j + 1] == vs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])


def topk(v, q=0.10):
    k = int(round(len(v) * q))
    return set(np.argsort(-v)[:k])


def size_floor(weeks, W):
    """HEAT rule: drop dongs whose 2023 mean is in the bottom 25%."""
    m = W[[i for i, w in enumerate(weeks) if w.year == 2023]].mean(axis=0)
    return m >= np.quantile(m, 0.25)


def anomaly_rows(weeks, start=date(2024, 8, 5), end=date(2024, 10, 27)):
    return np.array([start <= w <= end for w in weeks])
