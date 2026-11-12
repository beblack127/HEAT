"""out/_mapdata.json -> 가볍게 줄인 지도 데이터 -> 지도 도형 교체(mapgeo.py) -> 단독 HTML"""
import json, subprocess, sys
m = json.load(open("out/_mapdata.json"))
def rnd(c): return round(c, 5) if isinstance(c, (int, float)) else [rnd(x) for x in c]
for k in ["dongs", "gus", "gwon", "river"]:
    for f in m[k]["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
keep = ["dong_code8","tier","name_id","name","keyword_group_id","share","weight","is_primary","usable","share_500","share_1500","fnb_share_2325","method","evidence"]
rows = [{k: r.get(k, "") for k in keep} for r in m["links"]]
m["links"] = {"cols": keep, "rows": [[r[k] for k in keep] for r in rows]}
m["flags"] = {k: v for k, v in m["flags"].items() if v.get("flags") or v.get("note")}
json.dump(m, open("out/_mapdata_small.json", "w"), ensure_ascii=False, separators=(",", ":"))
subprocess.run([sys.executable, "mapgeo.py"], check=True)
# 지도 템플릿(v4: app_v4.html + Leaflet)에 지도 데이터와 랜드마크를 넣는다. 검색량이 없으면 이름표 화면만 보인다
# 사용: python3 make_html.py [template_v4.html] [landmarks.json]
tpl = sys.argv[1] if len(sys.argv) > 1 else "template_v4.html"
m = json.load(open("out/_mapdata_small.json"))
if len(sys.argv) > 2: m["landmarks"] = json.load(open(sys.argv[2], encoding="utf-8"))
data = json.dumps(m, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
open("out/행정동_이름표_확인지도.html", "w").write(open(tpl, encoding="utf-8").read().replace("__DATA__", data))
