"""데이터 매핑 과정 그림: 서교동 예시 3칸 (동네·역·상권). 출력: reports/gt_midterm/slides/fig/S10_mapping.png"""
import json
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import shape, Point
import matplotlib.patheffects as pe
src = open("11_slide_figs.py", encoding="utf-8").read(); exec(compile(src[:src.index("# ---- S1")], "h", "exec"))
KM = 5179; G = C.ROOT / "data/external/geo"
m = json.load(open(C.ROOT / "data/processed/nametags/_mapdata_small.json", encoding="utf-8"))
L = pd.read_csv(C.ROOT / "data/processed/nametags/04_행정동_이름표_연결표.csv", encoding="utf-8-sig", dtype={"dong_code8": str})
DN = "서교동"; code = L[L.dong_name == DN].dong_code8.iloc[0]; Ld = L[(L.dong_code8 == code) & (L.usable.astype(str) == "1")]
d = gpd.read_parquet(G / "emd_20251231.parquet").set_geometry("geom"); d = d[d.sidocd.astype(str) == "11"].to_crs(KM)
river = gpd.read_parquet(G / "han_river_5179.parquet").geometry.iloc[0]
dong = d[d.emdcd.astype(str).str[:8] == code].geometry.iloc[0].difference(river)
nb = d[d.geometry.intersects(dong.buffer(2500))]
umd = gpd.read_file(G / "umd.shp", encoding="cp949").to_crs(KM); umd["geometry"] = umd.geometry.buffer(0)
gw = gpd.GeoDataFrame.from_features(m["gwon"]["features"], crs=4326).to_crs(KM)
st = pd.DataFrame(m["stations"]); st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(KM)
EN = {"서교동": "Seogyo", "동교동": "Donggyo", "홍대입구": "Hongik Univ.", "합정": "Hapjeong", "상수": "Sangsu", "망원": "Mangwon", "서강대": "Sogang Univ.",
      "광흥창": "Gwangheungchang", "신촌": "Sinchon", "홍대": "Hongdae", "경의선책거리": "Gyeongui Line Book St.", "홍대땡땡거리": "Hongdae Ttaengttaeng St.", "신촌로터리": "Sinchon Rotary", "노고산동": "Nogosan"}
cx, cy = dong.centroid.x, dong.centroid.y; R = 2300
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
for r in pieces.itertuples():
    nm = r.EMD_NM; kept = nm in set(Ld[Ld.tier == "동네"].name)
    gpd.GeoSeries([r.geometry]).plot(ax=a, color=pal.get(nm, "#D9D9D9"), alpha=.75 if kept else .9, lw=0)
    p = r.geometry.representative_point()
    if r.sh >= .1: a.text(p.x, p.y, f"{EN.get(nm, nm)}\n{r.sh*100:.0f}%", color=INK, **TXT)
    elif r.sh >= .02: a.annotate(f"{EN.get(nm, nm)} {r.sh*100:.0f}%\n< 10%: dropped", xy=(p.x, p.y), xytext=(p.x + 300, p.y - 1100), color=GREY, arrowprops=dict(arrowstyle="->", color=GREY), **TXT)
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
    if len(s): pt = s.geometry.iloc[0]; off = {"홍대입구": (150, 230), "망원": (-200, -190)}.get(r.name, (0, -190)); a.text(pt.x + off[0], pt.y + off[1], f"{EN.get(r.name, r.name)} {float(r.weight)*100:.0f}%", color=INK, **TXT)
# 3 상권
a = axs[2]; lg = Ld[Ld.tier == "권역"]; cols = [RED, "#D98A6E", "#E8B3A0", "#F1CDBF"]
for (r, c) in zip(lg.itertuples(), cols):
    g = gw[gw.gwon == r.name]
    if not len(g): continue
    g.plot(ax=a, color=c, alpha=.55, lw=0); g.boundary.plot(ax=a, color=c, lw=1.2)
    p = g.geometry.iloc[0].intersection(dong).representative_point() if g.geometry.iloc[0].intersects(dong) else g.geometry.iloc[0].representative_point()
    if r.name == "홍대": a.text(p.x - 250, p.y - 150, f"{EN.get(r.name, r.name)} {float(r.weight)*100:.0f}%", color=INK, **TXT)
    elif float(r.weight) > .1: a.annotate(f"{EN.get(r.name, r.name)} {float(r.weight)*100:.0f}%", xy=(p.x, p.y), xytext=(p.x + 200, p.y - 1150), arrowprops=dict(arrowstyle="->", color=INK), color=INK, **TXT)
gpd.GeoSeries([dong]).boundary.plot(ax=axs[2], color=INK, lw=2.4)
f.tight_layout(w_pad=1); save(f, "S10_mapping.png")
print(pieces[["EMD_NM", "sh"]].round(3).to_string()); print(gw.columns.tolist()[:5])
