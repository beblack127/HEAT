"""데이터 매핑 과정 그림: 예시 동 3칸 (동네·역·상권). 기본 사근동 (2026-11-16 변경, 이전 서교동). 출력: reports/gt_midterm/slides/fig/S10_mapping.png"""
import json
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import shape, Point
import matplotlib.patheffects as pe
src = open("11_slide_figs.py", encoding="utf-8").read(); exec(compile(src[:src.index("# ---- S1")], "h", "exec"))
KM = 5179; G = C.ROOT / "data/external/geo"
m = json.load(open(C.ROOT / "data/processed/nametags/_mapdata_small.json", encoding="utf-8"))
L = pd.read_csv(C.ROOT / "data/processed/nametags/04_행정동_이름표_연결표.csv", encoding="utf-8-sig", dtype={"dong_code8": str})
import sys
DN = sys.argv[1] if len(sys.argv) > 1 else "사근동"; code = L[L.dong_name == DN].dong_code8.iloc[0]; Ld = L[(L.dong_code8 == code) & (L.usable.astype(str) == "1")]
d = gpd.read_parquet(G / "emd_20251231.parquet").set_geometry("geom"); d = d[d.sidocd.astype(str) == "11"].to_crs(KM)
river = gpd.read_parquet(G / "han_river_5179.parquet").geometry.iloc[0]
dong = d[d.emdcd.astype(str).str[:8] == code].geometry.iloc[0].difference(river)
nb = d[d.geometry.intersects(dong.buffer(2500))]
umd = gpd.read_file(G / "umd.shp", encoding="cp949").to_crs(KM); umd["geometry"] = umd.geometry.buffer(0)
gw = gpd.GeoDataFrame.from_features(m["gwon"]["features"], crs=4326).to_crs(KM)
st = pd.DataFrame(m["stations"]); st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(KM)
EN = {"서교동": "Seogyo", "동교동": "Donggyo", "홍대입구": "Hongik Univ.", "합정": "Hapjeong", "상수": "Sangsu", "망원": "Mangwon", "서강대": "Sogang Univ.",
      "광흥창": "Gwangheungchang", "신촌": "Sinchon", "홍대": "Hongdae", "경의선책거리": "Gyeongui Line Book St.", "홍대땡땡거리": "Hongdae Ttaengttaeng St.", "신촌로터리": "Sinchon Rotary", "노고산동": "Nogosan",
      "사근동": "Sageun", "행당동": "Haengdang", "한양대": "Hanyang Univ.", "용답": "Yongdap", "왕십리": "Wangsimni", "마장": "Majang", "답십리": "Dapsimni", "뚝섬": "Ttukseom",
      "성수동1가": "Seongsu", "용답동": "Yongdap", "마장동": "Majang", "응봉동": "Eungbong", "서울숲카페거리": "Seoul Forest Café St."}
cx, cy = dong.centroid.x, dong.centroid.y; R = 2300 if DN == "서교동" else 1700
f, axs = plt.subplots(1, 3, figsize=(18, 6.6))
TXT = dict(fontsize=18, fontweight="bold", ha="center", va="center", path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
for ax in axs:
    nb.boundary.plot(ax=ax, color="#C9CED6", lw=.8)
    gpd.GeoSeries([dong]).boundary.plot(ax=ax, color=INK, lw=2.4)
    ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R * .75, cy + R * .75); ax.set_aspect("equal"); ax.set_axis_off()
# 1 동네: 법정동 조각
a = axs[0]; pieces = gpd.overlay(gpd.GeoDataFrame(geometry=[dong], crs=KM), umd[["EMD_NM", "geometry"]], how="intersection")
pieces["sh"] = pieces.area / dong.area
pal = {"서교동": NAVY, "동교동": "#6E95C8"}
drop = []
for r in pieces.itertuples():
    nm = r.EMD_NM; kept = nm in set(Ld[Ld.tier == "동네"].name)
    gpd.GeoSeries([r.geometry]).plot(ax=a, color=(NAVY if r.sh == pieces.sh.max() else "#6E95C8") if kept else "#D0D3D8", alpha=.8, lw=0)
    geo = r.geometry; geo = max(geo.geoms, key=lambda g: g.area) if geo.geom_type == "MultiPolygon" else geo
    p = geo.representative_point()
    if kept and DN == "서교동": a.text(p.x, p.y, f"{EN.get(nm, nm)}\n{r.sh*100:.0f}%", color=INK, **TXT)
    elif kept:
        vx = p.x - cx; tx = (dong.bounds[2] + 350) if vx >= 0 else (dong.bounds[0] - 350)
        a.annotate(f"{EN.get(nm, nm)} {r.sh*100:.0f}%", xy=(p.x, p.y), xytext=(tx, p.y + 300), arrowprops=dict(arrowstyle="-", color=INK, lw=.8), color=INK, **{**TXT, "ha": "left" if vx >= 0 else "right"})
    elif r.sh >= .01: drop.append(f"{EN.get(nm, nm)} {r.sh*100:.0f}%")
if drop:
    bx = dong.bounds; a.text(bx[2] + 150, bx[1] - 250, "\n".join(drop) + "\n< 10%: dropped", color=GREY, **{**TXT, "ha": "left"})
# 2 역: 1,000m 원
a = axs[1]; gpd.GeoSeries([dong]).plot(ax=a, color="#EEF2F8", lw=0)
ls = Ld[Ld.tier == "역"]
for r in ls.itertuples():
    s = st[st.name == r.name]
    if not len(s): continue
    pt = s.geometry.iloc[0]; circ = pt.buffer(1000)
    gpd.GeoSeries([circ]).boundary.plot(ax=a, color=NAVY, lw=1, alpha=.45)
    gpd.GeoSeries([circ.intersection(dong)]).plot(ax=a, color=NAVY, alpha=.07, lw=0)
    a.plot(pt.x, pt.y, "o", color=RED, ms=7)
for r in ls.head(4).itertuples():
    s = st[st.name == r.name]
    if len(s):
        pt = s.geometry.iloc[0]; vx, vy = pt.x - cx, pt.y - cy; nv = max((vx**2 + vy**2) ** .5, 1)
        tx, ty = pt.x + vx / nv * 420, pt.y + vy / nv * 420
        a.annotate(f"{EN.get(r.name, r.name)} {float(r.weight)*100:.0f}%", xy=(pt.x, pt.y), xytext=(tx, ty), arrowprops=dict(arrowstyle="-", color=INK, lw=.8), color=INK, **{**TXT, "ha": "left" if vx > 0 else "right"})
# 3 상권
a = axs[2]; lg = Ld[Ld.tier == "권역"]; cols = [RED, "#D98A6E", "#E8B3A0", "#F1CDBF"]
for (r, c) in zip(lg.itertuples(), cols):
    g = gw[gw.gwon == r.name]
    if not len(g): continue
    g.plot(ax=a, color=c, alpha=.55, lw=0); g.boundary.plot(ax=a, color=c, lw=1.2)
    p = g.geometry.iloc[0].intersection(dong).representative_point()
    if float(r.weight) > .1: a.text(p.x, p.y, f"{EN.get(r.name, r.name)} {float(r.weight)*100:.0f}%", color=INK, **TXT)
if not len(lg):
    near = gw.assign(dist=gw.distance(dong)).sort_values("dist").head(1)
    near.plot(ax=a, color="#E8B3A0", alpha=.5, lw=0); near.boundary.plot(ax=a, color="#D98A6E", lw=1.2)
    p = near.geometry.iloc[0].representative_point()
    a.text(p.x, p.y - 300, f"{EN.get(near.gwon.iloc[0], near.gwon.iloc[0])}\noutside the dong", color=GREY, **TXT)
    c0 = dong.representative_point(); a.text(c0.x, c0.y, "no commercial area\n→ slot left empty", color=INK, **TXT)
gpd.GeoSeries([dong]).boundary.plot(ax=axs[2], color=INK, lw=2.4)
f.tight_layout(w_pad=1); save(f, "S10_mapping.png")
print(pieces[["EMD_NM", "sh"]].round(3).to_string()); print(gw.columns.tolist()[:5])
