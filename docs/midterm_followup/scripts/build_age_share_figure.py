"""Appendix figure: share of visitors aged 20-39 (purposes 4+5+7), hot places vs the rest of Seoul.

Uses 2025 monthly files (Jan-Dec as available), summed per dong. Standard library only.

Usage: python build_age_share_figure.py <codex_root> <out_dir>
"""
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

ROOT, OUT = sys.argv[1], sys.argv[2]
young, total = defaultdict(float), defaultdict(float)
for path in glob.glob(os.path.join(ROOT, "outputs", "nameplate_mobility_pilot", "age2030_mobility", "age2030_457_2025*.csv")):
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            young[r["admin_dong_code"]] += float(r["mobility_457_age20_39_daily"])
            total[r["admin_dong_code"]] += float(r["mobility_457_allages_daily"])
share = {d: young[d] / total[d] for d in total if total[d] > 0}
top = sorted(total, key=total.get, reverse=True)[: len(total) // 10]  # top 10% by all-age visits (not by 20-39)
rest = [share[d] for d in share if d not in top]
named = [("11440660", "Seogyo-dong (Hongdae)"), ("11680640", "Yeoksam 1-dong (Gangnam)"),
         ("11560540", "Yeoui-dong"), ("11110615", "Jongno 1-4ga-dong"), ("11140550", "Myeong-dong")]

rows = [(lbl, share[c], "hot") for c, lbl in named]
rows += [(f"Top 10% busiest dongs (median, n={len(top)})", st.median(share[d] for d in top), "group")]
seoul_median = st.median(share.values())

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "age_share_2025.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "share_age20_39"])
    for lbl, v, _ in rows:
        w.writerow([lbl, f"{v:.4f}"])
    w.writerow([f"Seoul median (all {len(share)} dongs)", f"{seoul_median:.4f}"])
    w.writerow([f"Other dongs median (n={len(rest)})", f"{st.median(rest):.4f}"])

# ---------- SVG ----------
W, H = 1600, 900
L, R, T = 560, 140, 170
BAR, GAP = 64, 30
XMAX = 0.70
PW = W - L - R


def x(v):
    return L + PW * v / XMAX


FONT = "font-family='Georgia, \"Times New Roman\", serif'"
COL = {"hot": "#1F4E9A", "group": "#5B8FD9"}
s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
     f"<rect width='{W}' height='{H}' fill='#F7F7F7'/>",
     f"<text x='80' y='70' {FONT} font-size='38' font-weight='bold' fill='#222'>Hot places draw a younger crowd</text>",
     f"<text x='80' y='112' {FONT} font-size='24' fill='#555'>Share of visitors aged 20–39 (shopping, tourism and other trips), 2025</text>"]
plot_h = len(rows) * (BAR + GAP) - GAP
for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    s.append(f"<line x1='{x(v):.1f}' x2='{x(v):.1f}' y1='{T - 10}' y2='{T + plot_h + 10}' stroke='#DDD'/>")
    s.append(f"<text x='{x(v):.1f}' y='{T + plot_h + 48}' {FONT} font-size='22' text-anchor='middle' fill='#333'>{int(v * 100)}%</text>")
for i, (lbl, v, kind) in enumerate(rows):
    yy = T + i * (BAR + GAP) + (16 if kind != "hot" else 0)
    s.append(f"<rect x='{L}' y='{yy}' width='{x(v) - L:.1f}' height='{BAR}' fill='{COL[kind]}'/>")
    s.append(f"<text x='{L - 20}' y='{yy + BAR / 2 + 9}' {FONT} font-size='25' text-anchor='end' fill='#222'>{lbl}</text>")
    s.append(f"<text x='{x(v) + 12:.1f}' y='{yy + BAR / 2 + 10}' {FONT} font-size='27' font-weight='bold' fill='#222'>{v * 100:.0f}%</text>")
xm = x(seoul_median)
s.append(f"<line x1='{xm:.1f}' x2='{xm:.1f}' y1='{T - 30}' y2='{T + plot_h + 26}' stroke='#B5432A' stroke-width='3' stroke-dasharray='10 6'/>")
s.append(f"<text x='{xm + 10:.1f}' y='{T - 36}' {FONT} font-size='22' fill='#B5432A'>Seoul median {seoul_median * 100:.1f}%</text>")
foot = [f"Source: Seoul living mobility, purposes 4+5+7, arrivals by dong. All {len(share)} dongs in the data.",
        "'Busiest' = top 10% by visits of all ages, so the group is not selected on age."]
for j, line in enumerate(foot):
    s.append(f"<text x='80' y='{H - 52 + j * 26}' {FONT} font-size='18' fill='#666'>{line}</text>")
s.append("</svg>")
with open(os.path.join(OUT, "age_share_appendix.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(s))
for r in rows:
    print(r[0], round(r[1], 3))
print("seoul median", round(seoul_median, 3), "dongs", len(share))
