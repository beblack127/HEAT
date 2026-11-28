"""Appendix figure: why Aug-Oct 2024 mobility is treated as a data issue.

Seoul-wide monthly totals, each shown as a ratio to the same month one year
earlier (1.0 = no change). Standard library only (no matplotlib available).

Usage: python build_anomaly_figure.py <codex_root> <heat_v3_quarterly_csv> <out_dir>
"""
import csv
import glob
import os
import sys
from collections import defaultdict

ROOT, CARD_CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MOB_ALL = os.path.join(ROOT, "work", "full_427_sources", "mobility_*.csv")
MOB_TGT = os.path.join(ROOT, "outputs", "nameplate_mobility_pilot", "age2030_mobility", "age2030_457_*.csv")
CTX = os.path.join(ROOT, "work", "Time-series-Project", "data", "context_monthly")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def monthly_sum(pattern, column):
    out = {}
    for path in sorted(glob.glob(pattern)):
        rows = read(path)
        out[rows[0]["month"]] = sum(float(r[column]) for r in rows)
    return out


series = {
    "target": monthly_sum(MOB_TGT, "mobility_457_age20_39_daily"),
    "mob_all": monthly_sum(MOB_ALL, "all_arrivals_daily"),
    "commute": monthly_sum(MOB_ALL, "purpose_1_daily"),
    "home": monthly_sum(MOB_ALL, "purpose_3_daily"),
    "living": monthly_sum(os.path.join(CTX, "living_*.csv"), "living_all_hour_mean"),
}
sub = defaultdict(float)
for r in read(os.path.join(CTX, "subway_station_monthly_2023_2025.csv")):
    sub[r["month"]] += float(r["exits"]) / float(r["days"])
series["subway"] = dict(sub)

# Card sales (F&B count) is quarterly and given as an index; convert to year-over-year.
card_q = {r["yq"]: float(r["food_cnt"]) for r in read(CARD_CSV)}


def prev_year(month):
    y, m = month.split("-")
    return f"{int(y) - 1}-{m}"


MONTHS = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 8)]
yoy = {k: {m: v[m] / v[prev_year(m)] for m in MONTHS if m in v and prev_year(m) in v} for k, v in series.items()}
card_yoy = {}
for yq, v in card_q.items():
    y, q = int(yq[:4]), int(yq[4])
    if f"{y - 1}{q}" in card_q and (y, q) >= (2024, 1) and (y, q) <= (2025, 2):
        card_yoy[(y, q)] = v / card_q[f"{y - 1}{q}"]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "anomaly_yoy_monthly.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["month"] + list(yoy))
    for m in MONTHS:
        w.writerow([m] + [f"{yoy[k][m]:.4f}" if m in yoy[k] else "" for k in yoy])
    for (y, q), v in sorted(card_yoy.items()):
        w.writerow([f"{y}Q{q} card_sales_fnb", f"{v:.4f}"])

# ---------- SVG ----------
W, H = 1600, 900
L, R, T, B = 140, 360, 140, 140
PW, PH = W - L - R, H - T - B
YMIN, YMAX = 0.7, 1.9
n = len(MONTHS)


def x(i):
    return L + PW * i / (n - 1)


def y(v):
    return T + PH * (YMAX - v) / (YMAX - YMIN)


STYLE = [  # key, label, color, width, dash
    ("target", "Mobility – our target (20–39, shop/tour/other)", "#1F4E9A", 5, ""),
    ("commute", "Mobility – commute only (all ages)", "#5B8FD9", 3, "10 6"),
    ("home", "Mobility – return home only (all ages)", "#9DBCE8", 3, "4 5"),
    ("living", "Living population", "#2E8B57", 3.5, ""),
    ("subway", "Subway exits", "#E08A1E", 3.5, ""),
]
FONT = "font-family='Georgia, \"Times New Roman\", serif'"
s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
     f"<rect width='{W}' height='{H}' fill='#F7F7F7'/>",
     f"<text x='{L}' y='62' {FONT} font-size='38' font-weight='bold' fill='#222'>Only mobility jumped in Aug–Oct 2024</text>",
     f"<text x='{L}' y='104' {FONT} font-size='24' fill='#555'>Seoul total, monthly, ratio to the same month one year earlier (1.0 = no change)</text>"]
i0, i1 = MONTHS.index("2024-08"), MONTHS.index("2024-10")
s.append(f"<rect x='{x(i0) - PW / (n - 1) / 2:.1f}' y='{T}' width='{(x(i1) - x(i0)) + PW / (n - 1):.1f}' height='{PH}' fill='#F6D3CB' opacity='0.8'/>")
for v in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
    s.append(f"<line x1='{L}' x2='{L + PW}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{'#888' if v == 1.0 else '#DDD'}' stroke-width='{2 if v == 1.0 else 1}'/>")
    s.append(f"<text x='{L - 16}' y='{y(v) + 8:.1f}' {FONT} font-size='24' text-anchor='end' fill='#333'>{v:.1f}</text>")
for i, m in enumerate(MONTHS):
    if m.endswith(("-01", "-04", "-07", "-10")):
        lab = ("Jan", "Apr", "Jul", "Oct")[("-01", "-04", "-07", "-10").index(m[4:])] + " " + m[:4]
        s.append(f"<text x='{x(i):.1f}' y='{T + PH + 40}' {FONT} font-size='24' text-anchor='middle' fill='#333'>{lab}</text>")
        s.append(f"<line x1='{x(i):.1f}' x2='{x(i):.1f}' y1='{T + PH}' y2='{T + PH + 8}' stroke='#555'/>")
s.append(f"<line x1='{L}' x2='{L + PW}' y1='{T + PH}' y2='{T + PH}' stroke='#555' stroke-width='1.5'/>")
s.append(f"<text transform='translate(52,{T + PH / 2}) rotate(-90)' {FONT} font-size='26' text-anchor='middle' fill='#333'>Ratio to same month last year</text>")

for key, label, color, width, dash in reversed(STYLE):
    pts = " ".join(f"{x(i):.1f},{y(yoy[key][m]):.1f}" for i, m in enumerate(MONTHS) if m in yoy[key])
    s.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{width}' stroke-dasharray='{dash}' stroke-linejoin='round'/>")
# card sales: quarterly points at the middle month of each quarter
cpts = []
for (yy, q), v in sorted(card_yoy.items()):
    mid = f"{yy}-{3 * q - 1:02d}"
    if mid in MONTHS:
        cpts.append((x(MONTHS.index(mid)), y(v)))
s.append(f"<polyline points='{' '.join(f'{a:.1f},{b:.1f}' for a, b in cpts)}' fill='none' stroke='#B5432A' stroke-width='3.5'/>")
for a, b in cpts:
    s.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='8' fill='#B5432A'/>")

peak = max(yoy["target"][m] for m in MONTHS[i0:i1 + 1])
s.append(f"<text x='{(x(i0) + x(i1)) / 2:.1f}' y='{T + 34}' {FONT} font-size='24' font-weight='bold' text-anchor='middle' fill='#B5432A'>Aug–Oct 2024</text>")

ly = T + 10
for key, label, color, width, dash in STYLE + [("card", "Card sales, F&amp;B (quarterly)", "#B5432A", 3.5, "")]:
    lx = L + PW + 30
    s.append(f"<line x1='{lx}' x2='{lx + 44}' y1='{ly}' y2='{ly}' stroke='{color}' stroke-width='{width}' stroke-dasharray='{dash}'/>")
    if key == "card":
        s.append(f"<circle cx='{lx + 22}' cy='{ly}' r='7' fill='{color}'/>")
    words, line = label.split(" "), ""
    rows = []
    for wd in words:
        if len(line + " " + wd) > 22:
            rows.append(line)
            line = wd
        else:
            line = (line + " " + wd).strip()
    rows.append(line)
    for j, row in enumerate(rows):
        s.append(f"<text x='{lx + 56}' y='{ly + 8 + j * 26}' {FONT} font-size='21' fill='#222'>{row}</text>")
    ly += 26 * len(rows) + 30

foot = ["Sources: Seoul living mobility (all 427 dongs), Seoul living population, subway station ridership, card sales (HEAT v3 table).",
        "2025 months are compared with 2024; the chart stops at Jul 2025 because later base months fall in the anomaly."]
for j, line in enumerate(foot):
    s.append(f"<text x='{L}' y='{H - 52 + j * 26}' {FONT} font-size='18' fill='#666'>{line}</text>")
s.append("</svg>")
with open(os.path.join(OUT, "anomaly_appendix.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(s))

for k in yoy:
    print(k, {m: round(yoy[k][m], 2) for m in ("2024-07", "2024-08", "2024-09", "2024-10", "2024-11") if m in yoy[k]})
print("card", {f"{a}Q{b}": round(v, 3) for (a, b), v in sorted(card_yoy.items())})
print("target peak", round(peak, 3))
