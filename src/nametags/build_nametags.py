"""행정동 이름표(검색어 신호 3칸) 연결표 생성 — v2 (위치 기준).

가중치는 시간에 따라 변하지 않는 기준으로 정한다(F&B 성장이 검색 가중치에 섞이거나 미래 정보가 들어가지 않게).
  - 동네 칸: 동의 육지 면적 중 법정동(브이월드 법정구역경계 2026-08)이 차지하는 비율
  - 역  칸: 동의 육지 면적(한강 수면 제외) 중 역에서 R m(기본 1,000m, 민감도 500/1,500m) 안에 드는 비율.
           원은 한강으로 잘라 역이 있는 쪽만 쓴다(강 건너 동에는 붙이지 않음)
  - 권역 칸: 권역 영역(+50m) 중 그 동에 걸친 비율이 10% 이상이면 연결, 가중치는 동 육지 면적 중 권역이 덮는 비율
음식점(2023~2025 영업) 기준 비율은 비교용 열(fnb_share_*)로만 남긴다.
"""
import re, json, sys, os
import numpy as np, pandas as pd, geopandas as gpd
from shapely.ops import unary_union

IN = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MIN_SHARE = {"동네": 0.10, "역": 0.05}
GWON_SIDE_MIN = 0.10  # 권역 면적 중 이 동에 걸친 비율
R_MAIN, R_SENS = 1000, (500, 1500)
KM = 5179  # 미터 단위 좌표계

# ---------------- 1. 행정동 ----------------
d = gpd.read_parquet(f"{IN}/emd_20251231.parquet").set_geometry("geom")
d = d[d.sidocd.astype(str) == "11"].copy().to_crs(KM)
d["dong_code10"] = d.emdcd.astype(str)
d["dong_code8"] = d.dong_code10.str[:8]
d = d.rename(columns={"sggnm": "gu", "emdnm": "dong_name"})[["dong_code10", "dong_code8", "gu", "dong_name", "geom"]]
d = d.rename_geometry("geometry")
d["area_km2"] = (d.area / 1e6).round(3)
river = gpd.read_parquet(f"{IN}/han_river_5179.parquet").geometry.iloc[0]
land = gpd.GeoDataFrame(d[["dong_code8"]].copy(), geometry=d.geometry.difference(river).values, crs=KM)
land["geometry"] = land.geometry.buffer(0)
land_area = land.set_index("dong_code8").area
d["land_km2"] = d.dong_code8.map(land_area / 1e6).round(3)

# ---------------- 2. 음식점 (2023~2025 중 영업한 적 있는 점포) ----------------
cols = ['source_type', 'business_type', 'lot_address', 'road_address', 'coordinate_x', 'coordinate_y',
        'permit_date', 'closure_date', 'status_normalized']
st = pd.read_csv(f"{IN}/stores.csv", usecols=cols, dtype=str, encoding="utf-8-sig")
pdt = pd.to_datetime(st.permit_date, errors="coerce"); cdt = pd.to_datetime(st.closure_date, errors="coerce")
bad = (st.status_normalized == "폐업") & cdt.isna()
st["act2325"] = ((pdt <= "2025-12-31") & (cdt.isna() | (cdt >= "2023-01-01")) & ~bad).values
st["act2301"] = ((pdt <= "2023-01-01") & (cdt.isna() | (cdt > "2023-01-01")) & ~bad).values  # 분석 시작 시점 스냅숏
st = st[st.act2325 | st.act2301].copy()
# 편의점 등 분석 제외 업태 (category_mapping.csv 기준)
EXCL = {"편의점", "키즈카페", "푸드트럭", "철도역구내", "공항", "고속도로", "유원지", "관광호텔", "백화점", "극장", "출장조리", "이동조리"}
st = st[~st.business_type.fillna("").isin(EXCL)]
st["x"] = pd.to_numeric(st.coordinate_x, errors="coerce"); st["y"] = pd.to_numeric(st.coordinate_y, errors="coerce")
n_all = int(st.act2325.sum()); n_all01 = int(st.act2301.sum())
st = st.dropna(subset=["x", "y"])
# 좌표계가 섞여 있다: 대부분 EPSG:5174(x 17~22만, y 43~47만), 일부 일반음식점은 EPSG:5179(x 94~98만, y 193~197만)
is5179 = (st.x > 800000) & (st.y > 1800000)
is5174 = st.x.between(150000, 260000) & st.y.between(400000, 500000)
stats0 = {"coord_epsg5174": int(is5174.sum()), "coord_epsg5179": int(is5179.sum()), "coord_unusable": int((~is5174 & ~is5179).sum())}
pa = gpd.GeoDataFrame(st[is5174], geometry=gpd.points_from_xy(st[is5174].x, st[is5174].y), crs=5174).to_crs(KM)
pb = gpd.GeoDataFrame(st[is5179], geometry=gpd.points_from_xy(st[is5179].x, st[is5179].y), crs=5179)
p = gpd.GeoDataFrame(pd.concat([pa, pb]), crs=KM)
p = gpd.sjoin(p, d[["dong_code8", "geometry"]], how="inner", predicate="within").drop(columns="index_right")
p01 = p[p.act2301].copy()
p = p[p.act2325].copy()
n_store = p.groupby("dong_code8").size()
n_store01 = p01.groupby("dong_code8").size()
d["n_fnb_2023_2025"] = d.dong_code8.map(n_store).fillna(0).astype(int)
d["n_fnb_2023_01"] = d.dong_code8.map(n_store01).fillna(0).astype(int)
stats = {"stores_active_2023_2025": n_all, "stores_active_2023_01": n_all01, **stats0,
         "stores_with_coords_in_seoul": int(len(p)), "stores_2023_01_in_seoul": int(len(p01))}

# ---------------- 3. 동네 칸: 법정동 ----------------
def legal_dong(addr):
    if not isinstance(addr, str): return None
    t = addr.split()
    if len(t) >= 3 and t[0].startswith("서울") and re.search(r"(동|가|로)(\d+가)?$", t[2]):
        return t[2]
    return None
p["legal_dong_raw"] = p.lot_address.map(legal_dong)
def legal_root_full(n):
    """성수동2가 -> 성수동, 종로1가 -> 종로, 을지로3가 -> 을지로"""
    if not isinstance(n, str): return None
    return re.sub(r"\d+가$", "", n)
p["legal_full"] = p.legal_dong_raw.map(legal_root_full)
p01["legal_full"] = p01.lot_address.map(legal_dong).map(legal_root_full)
stats["stores_with_legal_dong"] = int(p01.legal_full.notna().sum())

# v3: 법정동 경계(브이월드 LSMD_ADM_SECT_UMD, 2026-08)와 행정동 육지를 겹쳐 면적 비율로 정한다
umd = gpd.read_file(f"{IN}/umd.shp", encoding="cp949").to_crs(KM)
umd["geometry"] = umd.geometry.buffer(0)
umd["legal_full"] = umd.EMD_NM.map(legal_root_full)          # 성수동1가·2가 -> 성수동, 종로1가 -> 종로
ovl = gpd.overlay(land, umd[["legal_full", "geometry"]], how="intersection", keep_geom_type=True)
ld = ovl.assign(a=ovl.area).groupby(["dong_code8", "legal_full"]).a.sum().reset_index()
ld["store_share"] = ld.a / ld.dong_code8.map(land_area)     # 행정동 육지 면적 중 그 법정동 비율
ld["n"] = (ld.a / 1e6).round(3)                              # 동명이소 기준점 선택용(면적 ㎢)
ld = ld[ld.store_share >= MIN_SHARE["동네"]]
_l25 = p.dropna(subset=["legal_full"]).groupby(["dong_code8", "legal_full"]).size()
ld["fnb_share_2325"] = [(_l25.get((c, n), 0) / n_store.get(c, np.nan)) for c, n in zip(ld.dong_code8, ld.legal_full)]
# 민감도용: 2023-01-01 영업 음식점의 지번주소 법정동 비율(이전 기준). 이름 사전·검색어 묶음에는 함께 넣어 둔다
ld_alt = p01.dropna(subset=["legal_full"]).groupby(["dong_code8", "legal_full"]).size().rename("n").reset_index()
ld_alt["store_share"] = ld_alt.n / ld_alt.dong_code8.map(n_store01)
ld_alt = ld_alt[ld_alt.store_share >= MIN_SHARE["동네"]].copy()
ld_alt["gu"] = ld_alt.dong_code8.map(d.set_index("dong_code8").gu)
gu_of = d.set_index("dong_code8").gu
ld["gu"] = ld.dong_code8.map(gu_of)

def root_of(name):
    """검색어 뿌리: 끝의 '역'을 떼고, 3글자 이상이면 끝의 '동'을 뗀다. 서울역은 예외."""
    n = re.sub(r"역$", "", name)
    if n == "서울": return "서울역"
    if n.endswith("동") and len(n) >= 3: n = n[:-1]
    return n

# ---------------- 4. 역 칸 ----------------
s = pd.read_csv(f"{IN}/stations.csv", encoding="cp949")
s["main"] = s.역사명.str.replace(r"\(.*\)$", "", regex=True).str.strip()
s["sub"] = s.역사명.str.extract(r"\((.*)\)$")[0]
sg = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.경도, s.위도), crs=4326).to_crs(KM)
seoul = unary_union(d.geometry.values)
sg = sg[sg.distance(seoul) <= max(R_SENS)]
# 같은 이름의 역(환승역)은 좌표 평균으로 하나로 묶는다
sta = (sg.assign(px=sg.geometry.x, py=sg.geometry.y)
         .groupby("main").agg(px=("px", "mean"), py=("py", "mean"),
                              lines=("호선", lambda v: ", ".join(sorted(set(v)))),
                              subs=("sub", lambda v: ", ".join(sorted({x for x in v if isinstance(x, str)}))))
         .reset_index())
sta = gpd.GeoDataFrame(sta, geometry=gpd.points_from_xy(sta.px, sta.py), crs=KM)
sta["in_seoul"] = sta.within(seoul)

st_rows = []
for R in (R_MAIN,) + R_SENS:
    buf = sta[["main", "geometry"]].copy(); buf["geometry"] = buf.buffer(R)
    # 강 건너편은 그 역의 생활권이 아니다: 원을 한강으로 자른 뒤 역이 있는 쪽 조각만 남긴다
    def same_bank(circle, pt):
        g = circle.difference(river)
        if g.geom_type != "MultiPolygon": return g
        parts = sorted(g.geoms, key=lambda q: q.distance(pt))
        return parts[0]
    buf_bank = buf.copy(); buf_bank["geometry"] = [same_bank(c, pt) for c, pt in zip(buf.geometry, sta.geometry)]
    ov_ = gpd.overlay(land, buf_bank, how="intersection", keep_geom_type=True)
    a = ov_.assign(a=ov_.area).groupby(["dong_code8", "main"]).a.sum().rename(f"share_{R}") / 1.0
    j = gpd.sjoin(p[["dong_code8", "geometry"]], buf_bank, how="inner", predicate="within")
    c = j.groupby(["dong_code8", "main"]).size().rename(f"n_{R}")
    st_rows.append(pd.concat([a, c], axis=1).reset_index())
sl = st_rows[0]
for c in st_rows[1:]:
    sl = sl.merge(c, on=["dong_code8", "main"], how="outer")
sl = sl.fillna(0)
for R in (R_MAIN,) + R_SENS:
    sl[f"share_{R}"] = sl[f"share_{R}"] / sl.dong_code8.map(land_area)          # 면적 비율(주 기준)
    sl[f"fnb_share_{R}"] = sl[f"n_{R}"] / sl.dong_code8.map(n_store)            # 음식점 비율(비교용)
# 역–행정동 거리 (지하철 데이터 연결용)
dist = []
dd = d.set_index("dong_code8")
for _, r in sl.iterrows():
    dist.append(float(dd.loc[r.dong_code8, "geometry"].distance(sta.set_index("main").loc[r.main, "geometry"])))
sl["dist_to_dong_m"] = np.round(dist, 0)

# ---------------- 5. 권역 칸 ----------------
sk = gpd.read_file(f"{IN}/sk.shp").to_crs(KM)
sk_name = sk.TRDAR_CD_N.astype(str)
station_mains = set(sta.main)
FACILITY = re.compile(r"(마트|홈플러스|물산|아울렛|플라자|타워|점$|병원|구청|시청|우체국|법원|검찰|세무서|경찰|초등학교|중학교|고교|고등학교|아파트|교차로|사거리|오거리|삼거리|네거리|터미널|센터|공원|시장|빌딩|호텔|백화점|청사|주민센터|동주민|출구|\d)")
STREET = re.compile(r"(카페거리|카페골목|먹자골목|맛길|로데오|명품거리|가구거리|거리|골목|길|마을|파크)$")
TOUR = {  # 관광특구 6곳: 공식 명칭 -> 통칭 (정제 규칙, 수작업 아님이지만 표기 정리)
    "종로·청계 관광특구": "청계천", "동대문패션타운 관광특구": "동대문",
    "명동 남대문 북창동 다동 무교동 관광특구": "명동", "이태원 관광특구": "이태원",
    "강남 마이스 관광특구": "코엑스", "잠실 관광특구": "잠실",
}
cands = {}  # 권역명 -> set(상권 index)
GU_NAMES = set(d.gu)
def add(name, idx, how):
    name = name.strip()
    for g in GU_NAMES:  # '관악구 중앙길' -> '중앙길'은 너무 일반적이므로 구 이름을 붙인 채로 둔다. 띄어쓰기만 정리
        if name.startswith(g):
            name = name.replace(" ", ""); break
    if not name or len(name) < 2: return
    cands.setdefault(name, {"idx": set(), "how": set()})
    cands[name]["idx"].add(idx); cands[name]["how"].add(how)
for i, r in sk.iterrows():
    nm = str(r.TRDAR_CD_N); se = r.TRDAR_SE_C
    if se == "U":
        add(TOUR.get(nm, nm.replace(" 관광특구", "")), i, "관광특구"); continue
    main = re.sub(r"\(.*\)$", "", nm).strip()
    aliases = re.findall(r"\(([^)]*)\)", nm)
    alias_tokens = [a.strip() for al in aliases for a in re.split(r"[,，]", al)]
    if se == "D":
        for a in alias_tokens:
            a2 = re.sub(r"역$", "", a)
            if FACILITY.search(a) or a2 in station_mains or a in station_mains:
                continue
            add(a, i, "발달상권 별칭")
        if STREET.search(main) and not FACILITY.search(main):
            add(main, i, "발달상권 거리명")
    elif se in ("A", "R") and STREET.search(main) and not FACILITY.search(main) and se == "A":
        add(main, i, "골목상권 거리명")
# 법정동 이름과 같은 권역(인사동, 삼청동 등)은 동네 칸과 중복이므로 권역에서 제외
legal_names = set(ld.legal_full) | set(d.dong_name)
gw_rows, gw_geoms = [], []
for name, v in sorted(cands.items()):
    if name in legal_names or name + "동" in legal_names:
        continue
    geom = unary_union(sk.loc[sorted(v["idx"]), "geometry"].values).buffer(50)
    gw_rows.append({"gwon": name, "n_sangkwon": len(v["idx"]), "how": " + ".join(sorted(v["how"])),
                    "sangkwon_codes": " ".join(sk.loc[sorted(v["idx"]), "TRDAR_CD"].astype(str))})
    gw_geoms.append(geom)
gw = gpd.GeoDataFrame(gw_rows, geometry=gw_geoms, crs=KM)
ovg = gpd.overlay(land, gw[["gwon", "geometry"]], how="intersection", keep_geom_type=True)
gl = ovg.assign(a=ovg.area).groupby(["dong_code8", "gwon"]).a.sum().reset_index()
gl["gwon_side_share"] = gl.a / gl.groupby("gwon").a.transform("sum")        # 권역 중 이 동에 걸친 비율
gl["store_share"] = gl.a / gl.dong_code8.map(land_area)                     # 동 육지 면적 중 권역 비율(가중치용)
j = gpd.sjoin(p[["dong_code8", "geometry"]], gw[["gwon", "geometry"]], how="inner", predicate="within")
_g25 = j.groupby(["dong_code8", "gwon"]).size()
gl["n"] = [int(_g25.get((c, g), 0)) for c, g in zip(gl.dong_code8, gl.gwon)]
gl["fnb_share_2325"] = gl.n / gl.dong_code8.map(n_store)
gl = gl[gl.gwon_side_share >= GWON_SIDE_MIN]

# ---------------- 6. 이름 사전 + 검색어 묶음 ----------------
SUFFIXES = ("맛집", "카페", "술집", "데이트", "가볼만한곳", "핫플")  # 방문 의도 검색어(묶음 안에서 합산)
def kw(root, extra=()):
    base = [root] + [e for e in extra if e and e != root]
    out = []
    for b_ in base:
        for suf in SUFFIXES:
            out.append(f"{b_} {suf}")
    return list(dict.fromkeys(out))

def admin_root(n):
    """행정동명 -> 검색어 뿌리. 성수2가1동->성수, 역삼1동->역삼, 종로1·2·3·4가동->종로, 낙성대동->낙성대"""
    r = re.sub(r"[\d·]+가?[\d·]*동$", "", n)
    if r == n: r = re.sub(r"본동$", "", n)  # 면목본동 -> 면목
    # 숫자 동(역삼1동)은 '역삼'으로, 이름 자체가 다른 동(낙성대동, 대학동)은 '동'을 떼지 않는다('대학 맛집' 같은 일반어 방지)
    return r if r != n else n

# 6-1. 동네 칸 후보 (a) 법정동
dl = ld.copy()
dl["name"] = dl.legal_full; dl["root"] = dl.legal_full.map(root_of); dl["kind"] = "법정동"
# 6-1. 동네 칸 후보 (b) 행정동명이 법정동과 다를 때 (예: 관악구 낙성대동, 성동구 왕십리2동)
#   - 숫자를 뗀 이름(왕십리2동->왕십리, 목1동->목동)의 뿌리가 2km 안의 다른 이름표(역·법정동)와 같으면 그 묶음에 합친다
#   - 같은 뿌리가 2km 밖에만 있으면(관악구 신사동 vs 강남 신사역) 쓰지 않는다
#   - 아무 데도 없으면 '동'을 뗀지 않은 이름 그대로 쓴다(대학동 -> '대학 맛집' 같은 일반어 방지)
AMB_KM = 2000
dgeom = d.set_index("dong_code8").geometry
anchor_of = {}
for root_, g in dl.groupby("root"):
    anchor_of[root_] = dgeom[g.sort_values("n").dong_code8.iloc[-1]]
for m, g_ in zip(sta.main, sta.geometry):
    anchor_of[root_of(m)] = g_
def make_adm(own_roots):
    rows = []
    for r in d[["dong_code8", "gu", "dong_name"]].itertuples():
        base = admin_root(r.dong_name)
        if len(base) <= 1: base = base + "동"
        if base.endswith("제") and base[:-1] in own_roots.get(r.dong_code8, set()):  # 상일제1동 -> 상일동
            continue
        rr = root_of(base)
        if rr in own_roots.get(r.dong_code8, set()) or base in own_roots.get(r.dong_code8, set()):
            continue
        if rr in anchor_of:
            if dgeom[r.dong_code8].distance(anchor_of[rr]) > AMB_KM:
                continue
            root_ = rr
        else:
            root_ = base
        rows.append(dict(dong_code8=r.dong_code8, gu=r.gu, name=r.dong_name, root=root_, store_share=1.0,
                         n=int(n_store01.get(r.dong_code8, 0)), kind="행정동명", fnb_share_2325=1.0))
    return pd.DataFrame(rows)
own_roots = dl.groupby("dong_code8").root.apply(set)
adm = make_adm(own_roots)
adm_alt = make_adm(ld_alt.assign(r=ld_alt.legal_full.map(root_of)).groupby("dong_code8").r.apply(set))
_c = ["dong_code8", "gu", "name", "root", "store_share", "n", "kind", "fnb_share_2325"]
dong_c = pd.concat([dl[_c], adm[_c]], ignore_index=True)

# 6-2. 역 칸 후보
sc = sl[sl[f"share_{R_MAIN}"] >= MIN_SHARE["역"]].copy()
sc["name"] = sc.main; sc["root"] = sc.main.map(root_of)
sc["gu"] = sc.dong_code8.map(gu_of)
# 6-3. 권역 칸 후보
gc = gl.copy(); gc["name"] = gc.gwon; gc["root"] = gc.gwon.str.replace(r"\s+", "", regex=True); gc["gu"] = gc.dong_code8.map(gu_of)

# 6-4. 이름 겹침(동명이소) 판정: 같은 뿌리가 서로 떨어진 곳을 가리키면 '모호'
#   - 뿌리별로 연결된 행정동의 구 집합을 본다. 구가 둘 이상이고, 그 뿌리를 가진 역이 있으면 역이 있는 구를,
#     없으면 음식점이 더 많은 구를 '대표 지역'으로 두고, 나머지 구의 연결은 쓰지 않는다(usable=0).
allc = pd.concat([dong_c.assign(tier="동네"), sc[["dong_code8", "gu", "name", "root"]].assign(tier="역", n=sc[f"share_{R_MAIN}"]),
                  gc[["dong_code8", "gu", "name", "root", "n"]].assign(tier="권역")], ignore_index=True)
_main_pairs = set(zip(dl.dong_code8, dl.legal_full))
alt_only = ld_alt[[(c, n) not in _main_pairs for c, n in zip(ld_alt.dong_code8, ld_alt.legal_full)]]
allc = pd.concat([allc, pd.DataFrame({"dong_code8": alt_only.dong_code8, "gu": alt_only.gu, "name": alt_only.legal_full,
                                      "root": alt_only.legal_full.map(root_of), "n": 0.0, "tier": "동네", "kind": "법정동"})], ignore_index=True)
_adm_extra = adm_alt[~adm_alt.name.isin(set(adm.name))]
allc = pd.concat([allc, _adm_extra.assign(tier="동네", n=0.0)[["dong_code8", "gu", "name", "root", "n", "tier", "kind"]]], ignore_index=True)
ALT_ONLY_NAMES = (set(alt_only.legal_full) - set(dl.legal_full)) | (set(_adm_extra.name) - set(dl.legal_full))
# 같은 뿌리라도 서로 가까우면(이웃 동) 같은 장소다. 기준점에서 2km 넘게 떨어진 동만 '다른 장소'로 본다.
#   기준점 = 같은 뿌리의 역이 있으면 그 역, 없으면 음식점이 가장 많은 연결 동
AMB_KM = 2000
sta_root_pt = {root_of(m): g for m, g in zip(sta.main, sta.geometry)}
dgeom = d.set_index("dong_code8").geometry
amb = {}
for root, g in allc.groupby("root"):
    if g.dong_code8.nunique() <= 1: continue
    anchor = sta_root_pt.get(root)
    if anchor is None:
        anchor = dgeom[g.groupby("dong_code8").n.sum().idxmax()]
    far = sorted({c for c in g.dong_code8 if dgeom[c].distance(anchor) > AMB_KM})
    if far:
        amb[root] = {"drop_dongs": far, "drop_desc": ", ".join(f"{gu_of[c]} {d.set_index('dong_code8').dong_name[c]}" for c in far)}

# 6-5. 이름 사전
names = []
for (tier, nm), g in allc.groupby(["tier", "name"]):
    root = g.root.iloc[0]; flags = []; note = ""
    if tier == "동네":
        kinds = set(allc[(allc.tier == "동네") & (allc.name == nm)].kind.dropna())
        if nm in ALT_ONLY_NAMES: flags.append("민감도 분석(음식점 기준)에서만 쓰는 이름")
        extra = [nm] if (nm != root and not re.search(r"\d", nm)) else []
        src = "법정동(브이월드 법정구역경계 2026-08)" if "법정동" in kinds else "행정동명(법정동과 다른 이름)"
        if "법정동" not in kinds: flags.append("행정동명 기반(검색 사용 빈도 확인)")
        method = "규칙"
    elif tier == "역":
        r = sta.set_index("main").loc[nm]
        if root.endswith("역"): extra = []                       # 서울역
        elif nm != root: extra = [nm, nm + "역"]                # 둔촌동 -> 둔촌동, 둔촌동역
        else: extra = [root + "역"]
        src = f"역사마스터({r.lines})"; method = "규칙"
        if re.search(r"(시청|구청|법원|검찰|병원|대학교|공원|터미널|청사|운동장)", nm): flags.append("일반명사 포함(검색 의미 확인)")
        note = f"부기역명: {r.subs}" if r.subs else ""
    else:
        r = gw.set_index("gwon").loc[nm]
        extra = []; src = f"상권분석서비스 상권명({r.how})"; method = "규칙(상권명 추출)"
        flags.append("권역 검색어 확인 필요"); note = f"구성 상권 {r.n_sangkwon}개: {r.sangkwon_codes}"
    if len(root) <= 1: flags.append("한 글자 이름")
    if root in amb: flags.append(f"동명이소: 멀리 떨어진 {amb[root]['drop_desc']} 연결은 제외")
    names.append(dict(tier=tier, name=nm, root=root, keywords=kw(root, extra), source=src, method=method, flags=flags, note=note))
nd = pd.DataFrame(names)
# 같은 뿌리(root)는 하나의 검색어 묶음으로 합친다 (예: 성수동 + 성수역 -> '성수')
grp = (nd.groupby("root").agg(keywords=("keywords", lambda v: list(dict.fromkeys(k for ks in v for k in ks))),
                               tiers=("tier", lambda v: ", ".join(sorted(set(v), key=["동네", "역", "권역"].index))),
                               members=("name", lambda v: ", ".join(sorted(set(v)))))
         .reset_index().sort_values("root").reset_index(drop=True))
grp["keyword_group_id"] = [f"KG{i+1:04d}" for i in range(len(grp))]
grp["ambiguous"] = grp.root.isin(set(amb)).astype(int)
grp["ambiguity_rule"] = grp.root.map(lambda r: f"제외: {amb[r]['drop_desc']}" if r in amb else "")
grp["n_keywords"] = grp.keywords.map(len)
assert (grp.n_keywords <= 20).all(), grp[grp.n_keywords > 20]
# 한 검색어가 두 묶음에 들어가면 안 된다
_all = pd.Series([(k, r) for r, ks in zip(grp.root, grp.keywords) for k in ks]); _k = _all.map(lambda t: t[0]); assert not _k.duplicated().any(), _all[_k.duplicated(keep=False)].tolist()
grp["keywords"] = grp.keywords.map(lambda v: " | ".join(v))
nd = nd.merge(grp[["root", "keyword_group_id"]], on="root", how="left")
nd["flags"] = nd["flags"].map(lambda v: "; ".join(v) if v else "")
nd["keywords"] = nd["keywords"].map(lambda v: " | ".join(v))
TORD = {"동네": 0, "역": 1, "권역": 2}
nd = nd.sort_values(["tier", "name"], key=lambda c: c.map(TORD) if c.name == "tier" else c).reset_index(drop=True)
nd["name_id"] = [f"N{i+1:04d}" for i in range(len(nd))]
nid = {(r.tier, r.name): r.name_id for r in nd.itertuples()}
kgid = {(r.tier, r.name): r.keyword_group_id for r in nd.itertuples()}

# ---------------- 7. 행정동–이름표 연결표 ----------------
links = []
for r in dong_c.itertuples():
    links.append(dict(dong_code8=r.dong_code8, tier="동네", name=r.name, root=r.root, store_share=r.store_share,
                      fnb_share_2325=r.fnb_share_2325,
                      method="동 육지 면적 중 법정동 비율" if r.kind == "법정동" else "행정동명(동 전체=1)",
                      evidence=f"법정동 겹침 {r.n:.2f}㎢" if r.kind == "법정동" else "행정동 이름"))
for r in sc.itertuples():
    links.append(dict(dong_code8=r.dong_code8, tier="역", name=r.main, root=r.root, store_share=getattr(r, f"share_{R_MAIN}"),
                      fnb_share_2325=getattr(r, f"fnb_share_{R_MAIN}"), method=f"동 육지 면적 중 역 {R_MAIN}m 안 비율",
                      share_500=r.share_500, share_1500=r.share_1500, evidence=f"행정동 경계까지 {int(r.dist_to_dong_m)}m"))
for r in gc.itertuples():
    links.append(dict(dong_code8=r.dong_code8, tier="권역", name=r.gwon, root=r.root, store_share=r.store_share,
                      fnb_share_2325=r.fnb_share_2325, gwon_side_share=r.gwon_side_share,
                      method="동 육지 면적 중 권역 영역(+50m) 비율", evidence=f"권역 면적의 {r.gwon_side_share*100:.0f}%가 이 동에 있음"))
lk = pd.DataFrame(links)
lk["gu"] = lk.dong_code8.map(gu_of)
lk["usable"] = [0 if (rt in amb and c in amb[rt]["drop_dongs"]) else 1 for rt, c in zip(lk.root, lk.dong_code8)]
lk["name_id"] = [nid.get((t, n)) for t, n in zip(lk.tier, lk.name)]
lk["keyword_group_id"] = [kgid.get((t, n)) for t, n in zip(lk.tier, lk.name)]
u = lk.store_share.where(lk.usable == 1, 0.0)
lk["weight"] = u / u.groupby([lk.dong_code8, lk.tier]).transform("sum")
lk.loc[lk.usable == 0, "weight"] = 0.0
lk["is_primary"] = 0
# 대표 이름: 법정동 > 행정동명 순으로, 그 안에서 비율이 큰 것 (행정동명은 정의상 비율 1이라 우선순위를 낮춘다)
_pr = lk.assign(_k=(lk.method != "행정동명(동 전체=1)").astype(int))
idx = _pr[_pr.usable == 1].sort_values(["_k", "store_share"]).groupby(["dong_code8", "tier"]).tail(1).index
lk.loc[idx, "is_primary"] = 1
lk = lk.merge(d[["dong_code8", "dong_name"]], on="dong_code8", how="left")
for c in ("store_share", "weight", "share_500", "share_1500", "fnb_share_2325", "gwon_side_share"):
    if c in lk: lk[c] = lk[c].astype(float).round(4)
lk = lk[["dong_code8", "gu", "dong_name", "tier", "name_id", "name", "keyword_group_id", "store_share", "weight",
         "is_primary", "usable", "share_500", "share_1500", "gwon_side_share", "fnb_share_2325", "method", "evidence"]]
lk = lk.sort_values(["dong_code8", "tier", "store_share"], key=lambda c: c.map(TORD) if c.name == "tier" else c,
                    ascending=[True, True, False]).reset_index(drop=True)

# ---------------- 8. 행정동–역(전체), 행정동–상권 연결표 ----------------
dstat = sl.merge(sta[["main", "lines", "in_seoul"]], on="main", how="left").merge(d[["dong_code8", "gu", "dong_name"]], on="dong_code8")
dstat = dstat[(dstat.dist_to_dong_m <= max(R_SENS))]
dstat = dstat.rename(columns={"main": "station"})[["dong_code8", "gu", "dong_name", "station", "lines", "dist_to_dong_m",
                                                   "share_500", "share_1000", "share_1500", "fnb_share_500", "fnb_share_1000", "fnb_share_1500"]]
for c in ("share_500", "share_1000", "share_1500", "fnb_share_500", "fnb_share_1000", "fnb_share_1500"): dstat[c] = dstat[c].round(4)
dstat = dstat.sort_values(["dong_code8", "share_1000"], ascending=[True, False])

ov = gpd.overlay(sk[["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "ADSTRD_CD", "geometry"]], d[["dong_code8", "gu", "dong_name", "geometry"]], how="intersection", keep_geom_type=True)
ov["overlap_m2"] = ov.area
ov["sangkwon_share"] = ov.overlap_m2 / ov.TRDAR_CD.map(sk.set_index("TRDAR_CD").area)
ov = ov[ov.sangkwon_share >= 0.01]
dsk = pd.DataFrame(ov.drop(columns="geometry")).rename(columns={"TRDAR_CD": "sangkwon_code", "TRDAR_CD_N": "sangkwon_name",
                                                               "TRDAR_SE_1": "sangkwon_type", "ADSTRD_CD": "official_dong_code8"})
dsk["sangkwon_share"] = dsk.sangkwon_share.round(4); dsk["overlap_m2"] = dsk.overlap_m2.round(0)
dsk["is_official_dong"] = (dsk.official_dong_code8.astype(str) == dsk.dong_code8).astype(int)
dsk = dsk[["sangkwon_code", "sangkwon_name", "sangkwon_type", "dong_code8", "gu", "dong_name", "overlap_m2", "sangkwon_share",
           "official_dong_code8", "is_official_dong"]].sort_values(["sangkwon_code", "sangkwon_share"], ascending=[True, False])

# ---------------- 9. 행정동 목록 ----------------
cov = lk[lk.usable == 1].groupby(["dong_code8", "tier"]).size().unstack(fill_value=0).reindex(columns=["동네", "역", "권역"], fill_value=0)
dm = d.drop(columns="geometry").copy()
c4326 = d.geometry.representative_point().to_crs(4326)
dm["lat"] = c4326.y.round(6).values; dm["lon"] = c4326.x.round(6).values
for t in ("동네", "역", "권역"):
    dm[f"n_{t}"] = dm.dong_code8.map(cov[t]).fillna(0).astype(int)
dm["n_signal_slots"] = (dm[["n_동네", "n_역", "n_권역"]] > 0).sum(axis=1)
prim = lk[lk.is_primary == 1].pivot_table(index="dong_code8", columns="tier", values="name", aggfunc="first")
for t in ("동네", "역", "권역"):
    dm[f"primary_{t}"] = dm.dong_code8.map(prim[t]) if t in prim else ""
dm = dm.sort_values("dong_code8")

# ---------------- 10. 저장 ----------------
enc = "utf-8-sig"
dm.to_csv(f"{OUT}/01_행정동_목록.csv", index=False, encoding=enc)
grp[["keyword_group_id", "root", "keywords", "n_keywords", "tiers", "members", "ambiguous", "ambiguity_rule"]].to_csv(f"{OUT}/02_검색어_묶음.csv", index=False, encoding=enc)
nd[["name_id", "tier", "name", "keyword_group_id", "keywords", "source", "method", "flags", "note"]].to_csv(f"{OUT}/03_이름표_사전.csv", index=False, encoding=enc)
lk.rename(columns={"store_share": "share"}).to_csv(f"{OUT}/04_행정동_이름표_연결표.csv", index=False, encoding=enc)
# 07. 민감도: 동네 칸을 2023-01 음식점 지번주소 비율로 정했을 때 (역·권역은 본 결과와 같음)
alt = pd.concat([ld_alt.assign(name=ld_alt.legal_full, kind="법정동")[["dong_code8", "name", "store_share", "kind"]],
                 adm_alt.assign(kind="행정동명")[["dong_code8", "name", "store_share", "kind"]]], ignore_index=True)
alt["root"] = [root_of(n) if k == "법정동" else dict(zip(adm_alt.name, adm_alt.root)).get(n, n) for n, k in zip(alt.name, alt.kind)]
alt["usable"] = [0 if (rt in amb and c in amb[rt]["drop_dongs"]) else 1 for rt, c in zip(alt.root, alt.dong_code8)]
u_ = alt.store_share.where(alt.usable == 1, 0.0)
alt["weight"] = (u_ / u_.groupby(alt.dong_code8).transform("sum")).fillna(0).round(4)
alt["name_id"] = [nid.get(("동네", n)) for n in alt.name]
alt["keyword_group_id"] = [kgid.get(("동네", n)) for n in alt.name]
alt = alt.merge(d[["dong_code8", "gu", "dong_name"]], on="dong_code8")
alt["share"] = alt.store_share.round(4)
alt["basis"] = np.where(alt.kind == "법정동", "2023-01 음식점 지번주소 법정동 비율", "행정동명(동 전체=1)")
alt = alt[["dong_code8", "gu", "dong_name", "name_id", "name", "keyword_group_id", "share", "weight", "usable", "basis"]].sort_values(["dong_code8", "share"], ascending=[True, False])
alt.to_csv(f"{OUT}/07_민감도_동네칸_음식점기준.csv", index=False, encoding=enc)
assert alt.keyword_group_id.notna().all(), alt[alt.keyword_group_id.isna()].head()
dstat.to_csv(f"{OUT}/05_행정동_역_연결표.csv", index=False, encoding=enc)
dsk.to_csv(f"{OUT}/06_행정동_상권_연결표.csv", index=False, encoding=enc)

# 지도용 GeoJSON (단순화)
g4 = d[["dong_code8", "gu", "dong_name", "area_km2", "land_km2", "n_fnb_2023_2025"]].copy()
g4 = gpd.GeoDataFrame(g4, geometry=land.geometry.simplify(15).values, crs=KM).to_crs(4326)
gus = gpd.GeoDataFrame(d[["gu"]].copy(), geometry=land.geometry.values, crs=KM).dissolve("gu").reset_index()
gu_pts = gus.geometry.representative_point().to_crs(4326)
gus["geometry"] = gus.geometry.simplify(20); gus = gus.to_crs(4326)
riv4 = gpd.GeoSeries([river], crs=KM).to_crs(4326)
gw4 = gw[gw.gwon.isin(set(gl.gwon))].copy(); gw4["geometry"] = gw4.geometry.simplify(10); gw4 = gw4.to_crs(4326)
st4 = sta[sta.main.isin(set(sl.main))].to_crs(4326)
mapdata = {
    "dongs": json.loads(g4.to_json(drop_id=True)),
    "gwon": json.loads(gw4[["gwon", "n_sangkwon", "how", "geometry"]].to_json(drop_id=True)),
    "stations": [{"name": r.main, "lines": r.lines, "lat": round(r.geometry.y, 6), "lon": round(r.geometry.x, 6)} for r in st4.itertuples()],
    "links": lk.fillna("").to_dict(orient="records"),
    "groups": grp.set_index("keyword_group_id")[["root", "keywords", "ambiguous"]].to_dict(orient="index"),
    "flags": nd.set_index("name_id")[["flags", "note"]].to_dict(orient="index"),
    "gus": json.loads(gus[["gu", "geometry"]].to_json(drop_id=True)),
    "gu_labels": [{"gu": a, "lat": round(pt.y, 5), "lon": round(pt.x, 5)} for a, pt in zip(gus.gu, gu_pts)],
    "river": json.loads(riv4.to_json(drop_id=True)),
}
mapdata["links"] = [{("share" if k == "store_share" else k): v for k, v in r.items()} for r in mapdata["links"]]
for f in mapdata["dongs"]["features"]:
    for k, v in f["geometry"].items():
        pass
with open(f"{OUT}/_mapdata.json", "w", encoding="utf-8") as fh:
    json.dump(mapdata, fh, ensure_ascii=False, separators=(",", ":"))

# 요약 통계
stats.update({
    "n_dong": int(len(dm)), "n_names": {t: int((nd.tier == t).sum()) for t in ("동네", "역", "권역")},
    "n_keyword_groups": int(len(grp)), "n_ambiguous_groups": int(grp.ambiguous.sum()),
    "coverage": {t: int((dm[f"n_{t}"] > 0).sum()) for t in ("동네", "역", "권역")},
    "slots_dist": dm.n_signal_slots.value_counts().sort_index().to_dict(),
    "n_links": {t: int((lk.tier == t).sum()) for t in ("동네", "역", "권역")},
    "flagged_names": int((nd["flags"] != "").sum()), "links_unusable": int((lk.usable == 0).sum()), "n_admin_name_tags": int(len(adm)), "ambiguous_roots": {k: v["drop_desc"] for k, v in amb.items()},
    "dong_no_store": int((dm.n_fnb_2023_2025 == 0).sum()),
    "station_sensitivity": {
        str(R): int((sl[f"share_{R}"] >= MIN_SHARE["역"]).groupby(sl.dong_code8).any().sum()) for R in (500, 1000, 1500)},
})
with open(f"{OUT}/_stats.json", "w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=1, default=int)
print(json.dumps(stats, ensure_ascii=False, indent=1, default=int))
