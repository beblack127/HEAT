"""이름표 지도 + 검색량 레이어 (+ 개폐업 레이어) -> 단독 HTML.
사용: python3 make_search_html.py --mapdata _mapdata_small.json --template template_v4.html
        --dong 행정동_검색지수_월별.csv --groups 검색량_묶음별_월별.csv --quality 검색어묶음_품질.csv --out 지도.html
        [--fnb fnb_monthly.parquet --fnb-cats fnb_categories.json --rereg 07_hotspots_fnb_excluded_reregistration.csv]
개폐업: src/forecast/00_targets.py 가 만든 행정동 x 월 표. 부분 월(--fnb-end 다음 달부터)은 넣지 않는다.
"""
import argparse, json, math
import pandas as pd
ap = argparse.ArgumentParser()
for k in ("mapdata", "template", "dong", "groups", "quality", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--anchor", default="강남역 맛집")
ap.add_argument("--demo", default="")
ap.add_argument("--landmarks", default="")
ap.add_argument("--fnb", default="", help="개폐업 행정동 x 월 표 (fnb_monthly.parquet)")
ap.add_argument("--fnb-cats", default="", help="업종 이름 목록 (fnb_categories.json)")
ap.add_argument("--rereg", default="", help="재등록으로 보이는 동 목록 (07_hotspots_fnb_excluded_reregistration.csv)")
ap.add_argument("--fnb-start", default="2022-01", help="지도 시작 달. 검색량과 같게 2022-01")
ap.add_argument("--fnb-end", default="2026-09", help="마지막 완전한 달. 그 뒤 부분 월은 뺀다")
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
if a.fnb:
    fm = pd.read_parquet(a.fnb)
    fm = fm[(fm.month >= a.fnb_start) & (fm.month <= a.fnb_end)]
    fmonths = sorted(fm.month.unique())
    cats = json.load(open(a.fnb_cats, encoding="utf-8"))["categories"]
    kc = [f"Scat_{k:02d}" for k in range(len(cats))]
    ints = lambda s: [None if pd.isna(x) else int(round(x)) for x in s.tolist()]
    def series(g, with_cats):
        g = g.set_index("month").reindex(fmonths)
        o = {"O": ints(g.O), "Cd": ints(g.C_v_drop), "Ck": ints(g.C_v_keep), "S": ints(g.S)}
        if with_cats: o["K"] = [ints(g[c]) for c in kc]
        return o
    fdong = {c: series(g, True) for c, g in fm[fm.level == "dong"].groupby("series_id")}
    rereg = {}
    if a.rereg:
        for r in pd.read_csv(a.rereg, encoding="utf-8-sig", dtype={"dong_code8": str}).itertuples():
            rereg[r.dong_code8] = (f"같은 날 대량 개업({r.같은날10곳이상개업}곳)과 일괄 폐업({r.일괄폐업}곳)이 함께 있어 행정 재등록으로 보여요. "
                                   "순증률보다 점포 증감을 보세요.")
    m["fnb"] = {"months": fmonths, "cats": cats, "dong": fdong, "seoul": series(fm[fm.level == "seoul"], False), "rereg": rereg}
data = json.dumps(m, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
t = open(a.template, encoding="utf-8").read()
open(a.out, "w", encoding="utf-8").write(t.replace("__DATA__", data))
print("written", a.out, round(len(data.encode()) / 1e6, 2), "MB data")
