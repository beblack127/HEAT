"""지도용 도형: 행정동 원래 경계(강 포함), 구 경계, 한강 중심선(얇은 선)."""
import json, geopandas as gpd
from shapely.ops import unary_union, linemerge
from shapely.geometry import mapping
d = gpd.read_parquet("in/emd_20251231.parquet").set_geometry("geom"); d = d[d.sidocd.astype(str) == "11"].to_crs(5179)
d["dong_code8"] = d.emdcd.astype(str).str[:8]
river = gpd.read_parquet("in/han_river_5179.parquet").geometry.iloc[0]
N = unary_union(d[d.sggnm.isin(["마포구", "용산구", "성동구", "광진구"])].geometry.values)
S = unary_union(d[d.sggnm.isin(["강서구", "영등포구", "동작구", "서초구", "강남구", "송파구", "강동구"])].geometry.values)
seoul = unary_union(d.geometry.values)
mid = N.boundary.intersection(S.boundary)                       # 남북 구 경계 = 강 가운데
outer = seoul.boundary.intersection(river.buffer(-40))          # 서울 끝(강서·강동)에서 시 경계가 강 가운데를 지나는 구간
line = linemerge(unary_union([mid, outer]).intersection(river.buffer(30)))
line = line.simplify(25)
segs = line.geoms if hasattr(line, "geoms") else [line]
segs = [list(s.coords) for s in segs if s.length > 300]
# 조각을 서쪽 끝에서부터 가장 가까운 끝점 순으로 이어 하나의 선으로 만든다(시 경계가 강을 벗어나는 짧은 틈을 메움)
from shapely.geometry import LineString, Point
segs.sort(key=lambda c: min(c[0][0], c[-1][0]))
chain = segs.pop(0)
if chain[0][0] > chain[-1][0]: chain = chain[::-1]
while segs:
    end = Point(chain[-1])
    best = min(range(len(segs)), key=lambda i: min(end.distance(Point(segs[i][0])), end.distance(Point(segs[i][-1]))))
    c = segs.pop(best)
    if end.distance(Point(c[-1])) < end.distance(Point(c[0])): c = c[::-1]
    chain += c
segs = [LineString(chain).simplify(25)]
print("chained length", round(segs[0].length))
center = gpd.GeoSeries(segs, crs=5179).to_crs(4326)
def rnd(c): return round(c, 5) if isinstance(c, (int, float)) else [rnd(x) for x in c]
m = json.load(open("out/_mapdata_small.json"))
full = d[["dong_code8", "geom"]].copy(); full["geom"] = full.geometry.simplify(15); full = full.to_crs(4326).set_index("dong_code8").geometry
for f in m["dongs"]["features"]:
    f["geometry"] = mapping(full[f["properties"]["dong_code8"]]); f["geometry"]["coordinates"] = rnd(json.loads(json.dumps(f["geometry"]["coordinates"])))
gus = d.dissolve("sggnm").reset_index(); gus["geom"] = gus.geometry.simplify(20); gus = gus.to_crs(4326)
for f in m["gus"]["features"]:
    g = gus[gus.sggnm == f["properties"]["gu"]].geometry.iloc[0]
    f["geometry"] = json.loads(json.dumps(mapping(g))); f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
pts = gus.set_index("sggnm").geometry.to_crs(5179).representative_point().to_crs(4326)
m["gu_labels"] = [{"gu": k, "lat": round(p.y, 5), "lon": round(p.x, 5)} for k, p in pts.items()]
m["river"] = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": rnd([list(c) for c in s.coords])}} for s in center]}
json.dump(m, open("out/_mapdata_small.json", "w"), ensure_ascii=False, separators=(",", ":"))
