"""Does 20-39 mobility match card sales better than other ages? Level and growth, by dong."""
import csv, glob, io, os, struct, sys, zipfile
from collections import defaultdict

ROOT = sys.argv[1]
REF = os.path.join(ROOT, "work", "Time-series-Project", "data", "reference")

# --- commercial district -> dong (DBF) ---
z = zipfile.ZipFile(os.path.join(REF, "seoul_commercial_area_2023.zip"))
dbf = z.read([n for n in z.namelist() if n.endswith(".dbf")][0])
nrec, hlen, rlen = struct.unpack("<IHH", dbf[4:12])
fields, pos = [], 32
while dbf[pos] != 0x0D:
    name = dbf[pos:pos + 11].split(b"\0")[0].decode()
    fields.append((name, dbf[pos + 16]))
    pos += 32
tr2dong = {}
for i in range(nrec):
    rec, off, vals = dbf[hlen + i * rlen: hlen + (i + 1) * rlen], 1, {}
    for name, ln in fields:
        vals[name] = rec[off:off + ln].strip()
        off += ln
    tr2dong[vals["TRDAR_CD"].decode()] = vals["ADSTRD_CD"].decode()

# --- sales by dong, quarter, age ---
sales = defaultdict(lambda: defaultdict(float))  # (dong, yq) -> key -> amount
for yr in (2024, 2025):
    zz = zipfile.ZipFile(os.path.join(REF, f"seoul_commercial_sales_{yr}.zip"))
    txt = zz.read(zz.namelist()[0]).decode("cp949")
    for r in csv.DictReader(io.StringIO(txt)):
        d = tr2dong.get(r["상권_코드"])
        if not d:
            continue
        k = (d, r["기준_년분기_코드"])
        sales[k]["all"] += float(r["당월_매출_금액"])
        sales[k]["y2039"] += float(r["연령대_20_매출_금액"]) + float(r["연령대_30_매출_금액"])
print("quarters:", sorted({q for _, q in sales}))

# --- mobility (4+5+7) by dong and half-year ---
mob = defaultdict(lambda: defaultdict(float))
for p in glob.glob(os.path.join(ROOT, "outputs", "nameplate_mobility_pilot", "age2030_mobility", "age2030_457_*.csv")):
    ym = os.path.basename(p)[12:18]
    if ym[4:] > "06" or ym[:4] not in ("2024", "2025"):
        continue
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        k = (r["admin_dong_code"], ym[:4])
        y, a = float(r["mobility_457_age20_39_daily"]), float(r["mobility_457_allages_daily"])
        mob[k]["y2039"] += y; mob[k]["all"] += a; mob[k]["other"] += a - y


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def sales_h1(d, yr):
    return sum(sales[(d, f"{yr}{q}")]["all"] for q in (1, 2))


dongs = sorted({d for d, y in mob if y == "2025" and (d, "2024") in mob and sales_h1(d, 2025) > 0 and sales_h1(d, 2024) > 0})
print("dongs with sales:", len(dongs))
S25 = [sales_h1(d, 2025) for d in dongs]
print("\nLEVEL (2025 H1): Spearman with card sales")
for key, lab in (("y2039", "20-39 visits"), ("all", "all-age visits"), ("other", "non-20-39 visits")):
    print(f"  {lab:18s} {spearman([mob[(d, '2025')][key] for d in dongs], S25):.3f}")

print("\nGROWTH (2025 H1 / 2024 H1): Spearman with card sales growth")
G = [sales_h1(d, 2025) / sales_h1(d, 2024) for d in dongs]
for key, lab in (("y2039", "20-39 visits"), ("all", "all-age visits"), ("other", "non-20-39 visits")):
    g = [mob[(d, '2025')][key] / mob[(d, '2024')][key] if mob[(d, '2024')][key] > 0 else 1 for d in dongs]
    print(f"  {lab:18s} {spearman(g, G):.3f}")

# growth decomposition: top 10% dongs by all-age visit growth
allg = {d: mob[(d, '2025')]["all"] / mob[(d, '2024')]["all"] for d in dongs}
top = sorted(dongs, key=allg.get, reverse=True)[: len(dongs) // 10]
inc_y = sum(mob[(d, '2025')]["y2039"] - mob[(d, '2024')]["y2039"] for d in top)
inc_a = sum(mob[(d, '2025')]["all"] - mob[(d, '2024')]["all"] for d in top)
base = sum(mob[(d, '2024')]["y2039"] for d in top) / sum(mob[(d, '2024')]["all"] for d in top)
print(f"\nTop 10% growth dongs (n={len(top)}): 20-39 share of visits in 2024 {base:.3f}, share of the increase {inc_y / inc_a:.3f}")

# --- card sales age mix (independent source) ---
import statistics as st
def s25(d, key): return sum(sales[(d, f"2025{q}")][key] for q in (1, 2, 3, 4))
csh = {d: s25(d, "y2039") / s25(d, "all") for d in dongs if s25(d, "all") > 0}
msh = {d: mob[(d, '2025')]["y2039"] / mob[(d, '2025')]["all"] for d in csh}
print("\nCARD SALES age mix 2025")
print(f"  Spearman: 20-39 share of visits vs 20-39 share of card sales: {spearman(list(msh.values()), [csh[d] for d in msh]):.3f}")
busy = sorted(csh, key=lambda d: mob[(d, '2025')]["all"], reverse=True)[: len(csh) // 10]
print(f"  20-39 share of card sales: median all dongs {st.median(csh.values()):.3f}, busiest 10% {st.median(csh[d] for d in busy):.3f}")
for c, n in (("11440660", "Seogyo"), ("11680640", "Yeoksam1"), ("11560540", "Yeoui"), ("11110615", "Jongno1-4"), ("11140550", "Myeong")):
    print(f"  {n}: card 20-39 share {csh.get(c, float('nan')):.3f}")
g_sy = [sum(sales[(d, f'2025{q}')]['y2039'] for q in (1, 2)) / max(1, sum(sales[(d, f'2024{q}')]['y2039'] for q in (1, 2))) for d in dongs]
for key, lab in (("y2039", "20-39 visits"), ("other", "non-20-39 visits")):
    g = [mob[(d, '2025')][key] / mob[(d, '2024')][key] if mob[(d, '2024')][key] > 0 else 1 for d in dongs]
    print(f"  growth vs 20-39 card sales growth, {lab}: {spearman(g, g_sy):.3f}")
