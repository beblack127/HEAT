"""한강 수면 폴리곤 (OpenStreetMap, ODbL) -> 지도·면적 계산용으로 다듬기. 출력: in/han_river_5179.parquet"""
import json, sys
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
ts, emd, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(ts).read()
arr = json.loads(t[t.index("= [") + 2: t.rindex("]") + 1])
r = gpd.GeoSeries([MultiPolygon([Polygon(p[0], p[1:]) for p in arr]).buffer(0)], crs=4326).to_crs(5179).iloc[0]
d = gpd.read_parquet(emd).set_geometry("geom"); d = d[d.sidocd.astype(str) == "11"].to_crs(5179)
seoul = unary_union(d.geometry.values)
r = r.buffer(60).buffer(-60)                       # 교각·작은 틈 메우기
r = r.intersection(seoul.buffer(2500))             # 서울 밖은 2.5km까지만 (강이 지도 밖으로 흘러가는 모양)
def fill_small_holes(g, min_area=0.6e6):           # 밤섬·노들섬·선유도 같은 작은 섬은 물로 채움, 여의도는 남김
    polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
    out_ = []
    for p in polys:
        if p.area < 0.05e6: continue                # 떨어진 작은 물 조각 제거
        holes = [h for h in p.interiors if Polygon(h).area >= min_area]
        out_.append(Polygon(p.exterior, holes))
    return MultiPolygon(out_) if len(out_) > 1 else out_[0]
r = fill_small_holes(r)
# 샛강: OSM 한강 폴리곤에 빠져 있어, 여의동 경계 중 다른 영등포 동과 맞닿은 선을 따라 폭 120m 수로로 그린다(지도용 근사)
yi = d[d.emdnm == "여의동"].geometry.iloc[0]
others = unary_union(d[d.sggnm.isin(["영등포구", "동작구"]) & (d.emdnm != "여의동")].geometry.values)
saet = yi.boundary.intersection(others.buffer(5)).buffer(60, cap_style=2)
r = unary_union([r, saet]).buffer(20).buffer(-20).simplify(12)
gpd.GeoDataFrame({"name": ["한강"]}, geometry=[r], crs=5179).to_parquet(out)
print(r.geom_type, round(r.area / 1e6, 2), "km2", [round(Polygon(h).area/1e6,2) for p in (r.geoms if r.geom_type=='MultiPolygon' else [r]) for h in p.interiors])
