"""이름표 지도 + 검색량 레이어 -> 단독 HTML.
사용: python3 make_search_html.py --mapdata _mapdata_small.json --template template_search.html
        --dong 행정동_검색지수_월별.csv --groups 검색량_묶음별_월별.csv --quality 검색어묶음_품질.csv --out 지도.html [--demo "문구"]
"""
import argparse, json, math
import pandas as pd
ap = argparse.ArgumentParser()
for k in ("mapdata", "template", "dong", "groups", "quality", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--anchor", default="강남역 맛집")
ap.add_argument("--demo", default="")
ap.add_argument("--landmarks", default="")
a = ap.parse_args()

def r3(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return None
    if x == 0: return 0
    return float(f"{x:.3g}")

m = json.load(open(a.mapdata, encoding="utf-8"))
if a.landmarks: m["landmarks"] = json.load(open(a.landmarks, encoding="utf-8"))
ds = pd.read_csv(a.dong, encoding="utf-8-sig", dtype={"dong_code8": str})
gs = pd.read_csv(a.groups, encoding="utf-8-sig")
q = pd.read_csv(a.quality, encoding="utf-8-sig")
months = sorted(ds.month.unique())
dong = {}
for (c, t), g in ds.groupby(["dong_code8", "tier"]):
    v = g.set_index("month")["index"].reindex(months)
    dong.setdefault(c, {})[t] = [r3(x) for x in v.tolist()]
groups = {}
for k, g in gs.groupby("keyword_group_id"):
    v = g.set_index("month")["index"].reindex(months)
    groups[k] = [r3(x) for x in v.tolist()]
m["search"] = {"months": months, "anchor": a.anchor, "dong": dong, "groups": groups,
               "excluded": {r.keyword_group_id: r.reason for r in q[q.excluded == 1].itertuples()},
               "demo": a.demo}
data = json.dumps(m, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
t = open(a.template, encoding="utf-8").read()
open(a.out, "w", encoding="utf-8").write(t.replace("__DATA__", data))
print("written", a.out, round(len(data.encode()) / 1e6, 2), "MB data")
