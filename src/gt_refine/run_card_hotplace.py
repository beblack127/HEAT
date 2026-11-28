"""Rank dongs by young people's spending in hot-place business types, and check the ranking
against Seoul's official list of 121 major places.

Card sales: Seoul commercial-district estimated sales (OA-15572), 2023-2025, district -> dong
via the 2023 commercial-area file. Reference: Seoul's 121 major places (real-time city data),
assigned to dongs by area overlap (grid points inside each place polygon).

Usage: python run_card_hotplace.py <codex_root> <heat_root> <out_dir>
"""
import glob, io, json, os, struct, sys, zipfile
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT, HEAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REF = os.path.join(ROOT, "work", "Time-series-Project", "data", "reference")
os.makedirs(OUT, exist_ok=True)

# Hot-place business types: discretionary, experience-type spending that people travel for.
# Excluded: daily needs (supermarket, convenience store, clinics, academies, repairs) and
# everyday meals (Korean, Chinese, chicken, snack, fast food), which follow where people live or work.
HOT = {
    "food_drink": ["커피-음료", "제과점", "양식음식점", "일식음식점", "호프-간이주점"],
    "fashion_beauty": ["일반의류", "신발", "가방", "화장품", "시계및귀금속"],
    "leisure": ["노래방"],
}
HOT_ALL = [c for v in HOT.values() for c in v]


def read_dbf(raw, enc="utf-8"):
    n, h, r = struct.unpack("<IHH", raw[4:12]); fl, p = [], 32
    while raw[p] != 0x0D:
        fl.append((raw[p:p + 11].split(b"\0")[0].decode(), raw[p + 16])); p += 32
    out = []
    for i in range(n):
        rec, o, v = raw[h + i * r:h + (i + 1) * r], 1, {}
        for nm, ln in fl:
            v[nm] = rec[o:o + ln].decode(enc, "replace").strip(); o += ln
        out.append(v)
    return out


# ---------- card sales by dong ----------
z = zipfile.ZipFile(os.path.join(REF, "seoul_commercial_area_2023.zip"))
area = read_dbf(z.read([n for n in z.namelist() if n.endswith(".dbf")][0]), "cp949")
tr2dong = {int(a["TRDAR_CD"]): a["ADSTRD_CD"] for a in area}
frames = []
for y in range(2023, 2026):  # 2021-2022 left out (COVID period)
    zz = zipfile.ZipFile(os.path.join(REF, f"seoul_commercial_sales_{y}.zip"))
    frames.append(pd.read_csv(io.BytesIO(zz.read(zz.namelist()[0])), encoding="cp949"))
d = pd.concat(frames, ignore_index=True)
d["dong"] = d["상권_코드"].map(tr2dong)
d["year"] = d["기준_년분기_코드"] // 10
d["young"] = d["연령대_20_매출_금액"] + d["연령대_30_매출_금액"]
d["hot"] = d["서비스_업종_코드_명"].isin(HOT_ALL)
d["fnb"] = d["서비스_업종_코드_명"].isin(HOT["food_drink"] + HOT["leisure"])
d["young_cnt"] = d["연령대_20_매출_건수"] + d["연령대_30_매출_건수"]
unmapped = int(d.dong.isna().sum())
d = d.dropna(subset=["dong"])

idx = {
    "young_hot": d[d.hot].groupby(["year", "dong"]).young.sum(),
    "young_hot_cnt": d[d.hot].groupby(["year", "dong"]).young_cnt.sum(),       # transactions: less driven by luxury prices
    "young_fnb_cnt": d[d.fnb].groupby(["year", "dong"]).young_cnt.sum(),       # cafes, bakeries, western/Japanese, pubs, karaoke
    "all_hot": d[d.hot].groupby(["year", "dong"])["당월_매출_금액"].sum(),
    "young_all": d.groupby(["year", "dong"]).young.sum(),
    "all_sales": d.groupby(["year", "dong"])["당월_매출_금액"].sum(),
}
idx = {k: v.unstack().fillna(0) for k, v in idx.items()}

# mobility level for comparison (20-39, purposes 4+5+7), 2025 months available
mb = pd.concat([pd.read_csv(p, dtype={"admin_dong_code": str}) for p in
                glob.glob(os.path.join(ROOT, "outputs", "nameplate_mobility_pilot", "age2030_mobility", "age2030_457_2025*.csv"))])
mob25 = mb.groupby("admin_dong_code").mobility_457_age20_39_daily.mean()

names = pd.read_csv(os.path.join(HEAT, "data", "processed", "nametags", "01_행정동_목록.csv"), encoding="utf-8-sig", dtype={"dong_code8": str})
nm = dict(zip(names.dong_code8, names.gu + " " + names.dong_name))

# ---------- reference: Seoul 121 major places -> dongs ----------
def read_shp_polys(path):
    b = open(path, "rb").read(); pos, polys = 100, []
    while pos < len(b):
        _, ln = struct.unpack(">ii", b[pos:pos + 8]); rec = b[pos + 8:pos + 8 + ln * 2]; pos += 8 + ln * 2
        if struct.unpack("<i", rec[:4])[0] == 0:
            polys.append([]); continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
        pts = np.frombuffer(rec[44 + 4 * nparts:44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(-1, 2)
        polys.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return polys


def inside(px, py, rings):  # even-odd rule over all rings (holes handled)
    res = np.zeros(len(px), bool)
    for r in rings:
        x, y = r[:, 0], r[:, 1]; x2, y2 = np.roll(x, -1), np.roll(y, -1)
        for a, b_, c, e in zip(x, y, x2, y2):
            cond = (b_ > py) != (e > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = a + (py - b_) * (c - a) / (e - b_)
            res ^= cond & (px < xi)
    return res


base = glob.glob(os.path.join(REF, "seoul_121_place_boundaries", "*", "*.shp"))[0]
places = read_dbf(open(base[:-4] + ".dbf", "rb").read(), "utf-8")
ppoly = read_shp_polys(base)
gj = json.load(open(os.path.join(HEAT, "data", "processed", "nametags", "_mapdata_small.json"), encoding="utf-8"))["dongs"]
dpoly = {}
for f in gj["features"]:
    code = str(f["properties"]["dong_code8"])
    g = f["geometry"]; rings = []
    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        rings += [np.array(r, float) for r in poly]
    dpoly[code] = rings
dbbox = {k: (min(r[:, 0].min() for r in v), max(r[:, 0].max() for r in v), min(r[:, 1].min() for r in v), max(r[:, 1].max() for r in v)) for k, v in dpoly.items()}

place_dong = []
for p, rings in zip(places, ppoly):
    if not rings:
        continue
    allp = np.vstack(rings); x0, y0 = allp.min(0); x1, y1 = allp.max(0)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60)); gx, gy = gx.ravel(), gy.ravel()
    m = inside(gx, gy, rings); gx, gy = gx[m], gy[m]
    if len(gx) == 0:
        continue
    cnt = Counter()
    for code, (a, b_, c, e) in dbbox.items():
        if b_ < x0 or a > x1 or e < y0 or c > y1:  # bboxes do not overlap
            continue
        k = int(inside(gx, gy, dpoly[code]).sum())
        if k:
            cnt[code] = k
    tot = sum(cnt.values())
    for code, k in cnt.items():
        if k / tot >= 0.2 or k == max(cnt.values()):
            place_dong.append({"place": p["AREA_NM"], "category": p["CATEGORY"], "dong": code, "area_share": round(k / tot, 3)})
PD = pd.DataFrame(place_dong)
PD.to_csv(os.path.join(OUT, "place_to_dong.csv"), index=False, encoding="utf-8-sig")
TREND = ["발달상권", "관광특구"]
ref_trend = set(PD[PD.category.isin(TREND)].dong)
ref_station = set(PD[PD.category.eq("인구밀집지역")].dong)
trend_places = PD[PD.category.isin(TREND)].groupby("place").dong.apply(set).to_dict()

# ---------- evaluation ----------
universe = sorted(set(idx["young_hot"].columns) & set(mob25.index) & set(dpoly))
scores = {k: v.loc[2025].reindex(universe).fillna(0) for k, v in idx.items()}
scores["mobility_20_39_457"] = mob25.reindex(universe).fillna(0)
rows = []
for k, s in scores.items():
    for q in (0.05, 0.10, 0.15, 0.20):
        topk = set(s.nlargest(int(round(len(s) * q))).index)
        prec = len(topk & ref_trend) / len(topk)
        rec_places = np.mean([bool(ds & topk) for ds in trend_places.values()])
        rows.append({"index": k, "top": f"{int(q * 100)}%", "n": len(topk), "precision_trend": prec,
                     "recall_trend_places": rec_places, "precision_trend_or_station": len(topk & (ref_trend | ref_station)) / len(topk)})
EV = pd.DataFrame(rows)
EV.to_csv(os.path.join(OUT, "reference_agreement.csv"), index=False, float_format="%.3f")
base_rate = len(ref_trend & set(universe)) / len(universe)

# ---------- ranking lists ----------
s = scores["young_hot"]; tot25 = s.sum()
share = d[d.hot & d.year.eq(2025)].groupby("dong").young.sum() / d[d.hot & d.year.eq(2025)].groupby("dong")["당월_매출_금액"].sum()
top = s.nlargest(42)
places_by_dong = PD[PD.category.isin(TREND + ["인구밀집지역"])].groupby("dong").place.apply(lambda x: ", ".join(sorted(set(x))))
L = pd.DataFrame({"rank": range(1, 43), "dong": [nm.get(c, c) for c in top.index], "code": top.index,
                  "young_hot_sales_billion_krw": (top / 1e8).round(1), "share_of_seoul_pct": (top / tot25 * 100).round(2),
                  "young_share_in_hot_types_pct": (share.reindex(top.index) * 100).round(1),
                  "mobility_rank": scores["mobility_20_39_457"].rank(ascending=False).reindex(top.index).astype(int),
                  "official_places": places_by_dong.reindex(top.index).fillna("")})
L.to_csv(os.path.join(OUT, "top42_young_hot_2025.csv"), index=False, encoding="utf-8-sig")

# risers: 2023 -> 2025 change in share of Seoul young hot-type spending (size floor: 2023 median)
y23, y25 = idx["young_hot"].loc[2023].reindex(universe).fillna(0), idx["young_hot"].loc[2025].reindex(universe).fillna(0)
ok = y23 >= y23.median()
g = (np.log(y25[ok] / y25.sum()) - np.log(y23[ok] / y23.sum())).replace([np.inf, -np.inf], np.nan).dropna()
rz = g.nlargest(20)
RZ = pd.DataFrame({"dong": [nm.get(c, c) for c in rz.index], "code": rz.index, "share_change_pct": ((np.exp(rz) - 1) * 100).round(1),
                   "young_hot_2025_billion_krw": (y25[rz.index] / 1e8).round(1), "official_places": places_by_dong.reindex(rz.index).fillna("")})
RZ.to_csv(os.path.join(OUT, "risers_2023_2025.csv"), index=False, encoding="utf-8-sig")

summary = {"rows": len(d), "unmapped_rows": unmapped, "dongs_with_sales": int(d.dong.nunique()), "universe": len(universe),
           "hot_types": HOT, "trend_places": len(trend_places), "trend_dongs_in_universe": len(ref_trend & set(universe)),
           "base_rate_trend": round(base_rate, 3),
           "young_hot_share_of_all_sales_2025": round(float(d[d.hot & d.year.eq(2025)].young.sum() / d[d.year.eq(2025)]["당월_매출_금액"].sum()), 3)}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(json.dumps(summary, ensure_ascii=False))
print(EV.pivot(index="index", columns="top", values="precision_trend").round(2))
print(EV.pivot(index="index", columns="top", values="recall_trend_places").round(2))
print(L.head(30).to_string(index=False))
print(RZ.to_string(index=False))
