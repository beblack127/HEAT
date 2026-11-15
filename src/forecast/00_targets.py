"""대상 테이블 만들기.

검색: 행정동 × 칸 × 월 서울 대비 비중 + fold별 계절 계수 (학습 구간 자료만)
개폐업: 음식점(include == 1)을 좌표로 행정동에 붙이고 동 × 월 개업·폐업·영업 점포 수와 F&B 지표 재료를 만든다.
        서울·구 수준은 동 합계.

출력 (data/interim/forecast/)
  search_monthly.parquet     dong_code8, gu, tier, month, index, share
  search_seasonal.parquet    fold, dong_code8, tier, moy, sf
  stores_mapped.parquet      점포 단위 (행정동, 범주, 개업·폐업일, 일괄 폐업 플래그)
  fnb_monthly.parquet        level, series_id, gu, month, O, C_v_keep, C_v_drop, S, ... (2015-01 ~ 2026-10)
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import C, Timer, log_run, mrange, read_csv

C.INTERIM.mkdir(parents=True, exist_ok=True)
C.REPORTS.mkdir(parents=True, exist_ok=True)
stats = {}

# =============== 1. 행정동 ===============
dl = read_csv(C.DONG_LIST_CSV, dtype={"dong_code8": str, "dong_code10": str})
dong_gu = dl.set_index("dong_code8").gu
emd = gpd.read_parquet(C.EMD_PARQUET).set_geometry("geom")
emd = emd[emd.sidocd.astype(str) == "11"].to_crs(5179)
emd["dong_code8"] = emd.emdcd.astype(str).str[:8]
emd = emd.rename_geometry("geometry")[["dong_code8", "geometry"]]

# =============== 2. 검색 ===============
with Timer("검색 대상"):
    s = read_csv(C.SEARCH_CSV, dtype={"dong_code8": str})
    s = s[["dong_code8", "tier", "month", "index", "seoul_share"]].rename(columns={"seoul_share": "share"})
    s["gu"] = s.dong_code8.map(dong_gu)
    s.to_parquet(C.INTERIM / "search_monthly.parquet", index=False)

    # fold별 계절 계수: compute_dong_search.py와 같은 정의, 단 학습 구간(<= train_end) 자료만 사용
    def seasonal(v, end):
        v = v[v.index <= end]
        tr = v.rolling(12, min_periods=12).mean().shift(-6)
        r = (v / tr)[tr.notna()]
        sf = r.groupby(r.index.str[5:]).mean().reindex([f"{m:02d}" for m in range(1, 13)])
        if sf.notna().all() and sf.mean() > 0:
            return sf / sf.mean()
        return pd.Series(1.0, index=sf.index)

    rows = []
    for fold, end in [("F2024", "2023-12"), ("F2025", "2024-12"), ("FINAL", "2025-12")]:
        for (d, t), g in s.groupby(["dong_code8", "tier"]):
            sf = seasonal(g.set_index("month").share, end)
            rows.append(pd.DataFrame({"fold": fold, "dong_code8": d, "tier": t, "moy": sf.index, "sf": sf.values}))
    pd.concat(rows).to_parquet(C.INTERIM / "search_seasonal.parquet", index=False)
    stats["search_dongs_total"] = int(s[s.tier == "전체"].dong_code8.nunique())

# =============== 3. 음식점 점포 ===============
with Timer("음식점 읽기"):
    ev = read_csv(C.EVENTS_CSV, dtype=str,
                  usecols=["source_type", "management_id", "event_type", "event_date", "business_type"])
    st = read_csv(C.STORES_CSV, dtype=str,
                  usecols=["source_type", "management_id", "business_type", "business_name", "coordinate_x",
                           "coordinate_y", "permit_date", "closure_date", "status_normalized",
                           "last_modified_raw", "data_update_raw", "closed_status_missing_closure_date_flag"])
    st = st.drop_duplicates(["source_type", "management_id"])  # 휴게음식점 중복 관리번호 323개: 첫 행

    # 개업·폐업일은 사건 테이블 기준 (중복 제거 끝난 값)
    key = ["source_type", "management_id"]
    op = ev[ev.event_type == "개업"].drop_duplicates(key).set_index(key).event_date.rename("open_date")
    cl = ev[ev.event_type == "폐업"].drop_duplicates(key).set_index(key).event_date.rename("close_date")
    st = st.join(op, on=key).join(cl, on=key)
    stats["stores_raw"] = int(len(st))

# 업종 매핑
cm = read_csv(C.CATEGORY_CSV)
cm = cm[cm.source.isin(["일반음식점", "휴게음식점"])]
st["business_type"] = st.business_type.fillna("(결측)").str.strip()
st = st.merge(cm[["source", "raw_value", "category", "include", "is_cafe", "is_trend"]],
              left_on=["source_type", "business_type"], right_on=["source", "raw_value"], how="left")
unm = st[st.category.isna()].groupby(["source_type", "business_type"]).size().rename("n_stores").reset_index()
unm.to_csv(C.UNMAPPED_CSV, index=False, encoding="utf-8-sig")
stats["unmapped_categories"] = unm.to_dict("records")
stats["stores_unmapped_category"] = int(unm.n_stores.sum())
stats["stores_excluded_category"] = int((st.include == 0).sum())
st = st[st.include == 1].copy()

# 폐업 상태인데 폐업일이 없는 기록: 영업 기간을 알 수 없어 제외
bad = (st.status_normalized == "폐업") & st.close_date.isna()
stats["stores_closed_without_date"] = int(bad.sum())
st = st[~bad].copy()
st["open_date"] = pd.to_datetime(st.open_date, errors="coerce")
st["close_date"] = pd.to_datetime(st.close_date, errors="coerce")
st = st[st.open_date.notna()].copy()

# 좌표 -> 행정동 (EPSG:5174 / 5179 섞임)
with Timer("행정동 결합"):
    st["x"] = pd.to_numeric(st.coordinate_x, errors="coerce")
    st["y"] = pd.to_numeric(st.coordinate_y, errors="coerce")
    is5179 = (st.x > 800000) & (st.y > 1800000)
    is5174 = st.x.between(150000, 260000) & st.y.between(400000, 500000)
    stats["coord_epsg5174"] = int(is5174.sum())
    stats["coord_epsg5179"] = int(is5179.sum())
    stats["coord_unusable"] = int((~is5174 & ~is5179).sum())
    pa = gpd.GeoDataFrame(st[is5174], geometry=gpd.points_from_xy(st[is5174].x, st[is5174].y), crs=5174).to_crs(5179)
    pb = gpd.GeoDataFrame(st[is5179], geometry=gpd.points_from_xy(st[is5179].x, st[is5179].y), crs=5179)
    p = gpd.GeoDataFrame(pd.concat([pa, pb]), crs=5179)
    p = gpd.sjoin(p, emd, how="left", predicate="within").drop(columns="index_right")
    st = st.join(p["dong_code8"])
    # 좌표 없는 점포의 비중을 시기별로 남긴다 (01_checks에서 보고)
    st["open_year"] = st.open_date.dt.year
    miss = st.assign(miss=st.dong_code8.isna()).groupby("open_year").miss.agg(["mean", "size"])
    miss.to_csv(C.INTERIM / "dong_unmapped_by_year.csv", encoding="utf-8-sig")
    stats["stores_include"] = int(len(st))
    stats["stores_mapped_to_dong"] = int(st.dong_code8.notna().sum())
    st = st[st.dong_code8.notna()].copy()
    st["gu"] = st.dong_code8.map(dong_gu)

# 일괄 폐업 플래그: 같은 동·같은 폐업일 건수 >= 동 일평균 x 10 이면서 >= 10건
c = st[st.close_date.notna() & (st.close_date >= C.HIST_START)]
n_days = (c.close_date.max() - pd.Timestamp(C.HIST_START)).days + 1
daily_mean = c.groupby("dong_code8").size() / n_days
same_day = c.groupby(["dong_code8", "close_date"]).size().rename("n_same_day").reset_index()
same_day["thr"] = np.maximum(same_day.dong_code8.map(daily_mean) * C.BULK_MULT, C.BULK_MIN)
same_day["bulk"] = same_day.n_same_day >= same_day.thr
st = st.merge(same_day[["dong_code8", "close_date", "n_same_day", "bulk"]], on=["dong_code8", "close_date"], how="left")
st["bulk_closure"] = st.bulk.fillna(False).astype(bool)
st = st.drop(columns=["bulk"])
stats["bulk_closure_stores"] = int(st.bulk_closure.sum())
stats["bulk_closure_events"] = int(same_day.bulk.sum())

keep_cols = ["source_type", "management_id", "business_type", "business_name", "category", "is_cafe", "is_trend",
             "open_date", "close_date", "status_normalized", "last_modified_raw", "data_update_raw",
             "dong_code8", "gu", "bulk_closure", "n_same_day"]
st[keep_cols].to_parquet(C.INTERIM / "stores_mapped.parquet", index=False)

# =============== 4. 동 × 월 패널 ===============
with Timer("동 x 월 집계"):
    months = mrange(C.HIST_START, "2026-10")
    M = len(months)
    m0 = pd.Period(C.HIST_START, "M")
    dongs = sorted(dong_gu.index)
    di = {d: i for i, d in enumerate(dongs)}
    D = len(dongs)
    st["di"] = st.dong_code8.map(di)
    # 월 위치: -1 = HIST_START 이전 (재고에만 반영), M = 데이터 끝 이후(없음)
    base = m0.year * 12 + m0.month - 1
    om = np.clip((st.open_date.dt.year * 12 + st.open_date.dt.month - 1 - base).values, -1, None).astype(int)
    cmf = (st.close_date.dt.year * 12 + st.close_date.dt.month - 1 - base).values
    cm_ = np.clip(np.where(np.isnan(cmf), M + 10, cmf), -1, M + 10).astype(int)
    dd = st.di.values.astype(int)
    cats = sorted(st.category.unique())

    def acc(idx_m, weights=None, mask=None):
        """[D, M+1] 배열에 더한다. 열 0 = 시작 이전, 열 k+1 = months[k]."""
        a = np.zeros((D, M + 1))
        sel = (idx_m < M) if mask is None else ((idx_m < M) & mask)
        w = np.ones(sel.sum()) if weights is None else np.asarray(weights)[sel]
        np.add.at(a, (dd[sel], idx_m[sel] + 1), w)
        return a

    def stock(open_a, close_a):
        return (np.cumsum(open_a, 1) - np.cumsum(close_a, 1))[:, 1:]

    out = {}
    O = acc(om)
    Ck = acc(cm_)
    bulk = st.bulk_closure.values
    Cd = acc(cm_, mask=~bulk)
    out["O"] = O[:, 1:]
    out["C_v_keep"] = Ck[:, 1:]
    out["C_v_drop"] = Cd[:, 1:]
    out["bulk_n"] = (Ck - Cd)[:, 1:]
    out["S"] = stock(O, Ck)        # 영업 점포 수는 실제 등록 상태 (두 패널 공통)
    for flag in ("is_cafe", "is_trend"):
        msk = st[flag].values == 1
        out[f"S_{flag[3:]}"] = stock(acc(om, mask=msk), acc(cm_, mask=msk))
        out[f"O_{flag[3:]}"] = acc(om, mask=msk)[:, 1:]
    for k, cat in enumerate(cats):
        msk = st.category.values == cat
        out[f"Scat_{k:02d}"] = stock(acc(om, mask=msk), acc(cm_, mask=msk))
    # 업력: 영업 점포 개업일 합 (일 단위 ordinal)
    ordv = st.open_date.map(pd.Timestamp.toordinal).values.astype(float)
    out["age_ord_sum"] = stock(acc(om, weights=ordv), acc(cm_, weights=ordv))
    # 6개월 생존 코호트 재료: 개업월별 개업 수, 6개월 뒤 영업 중 수
    surv = (st.close_date.isna() | (st.close_date > st.open_date + pd.DateOffset(months=6))).values
    out["coh_n"] = O[:, 1:]
    out["coh_surv"] = acc(om, mask=surv)[:, 1:]
    # 폐업 점포 업력 (개월)
    ten = ((st.close_date - st.open_date).dt.days / 30.4375).fillna(0).values
    out["ten_sum"] = acc(cm_, weights=ten, mask=~bulk)[:, 1:]

    rows = []
    for d in dongs:
        i = di[d]
        df = pd.DataFrame({k: v[i] for k, v in out.items()})
        df.insert(0, "month", months)
        df.insert(0, "gu", dong_gu[d])
        df.insert(0, "series_id", d)
        rows.append(df)
    dong_p = pd.concat(rows, ignore_index=True)
    dong_p.insert(0, "level", "dong")
    num = [c for c in dong_p.columns if c not in ("level", "series_id", "gu", "month")]
    gu_p = dong_p.groupby(["gu", "month"])[num].sum().reset_index()
    gu_p["series_id"] = gu_p.gu
    gu_p.insert(0, "level", "gu")
    se_p = dong_p.groupby("month")[num].sum().reset_index()
    se_p["series_id"] = "서울"
    se_p["gu"] = "서울"
    se_p.insert(0, "level", "seoul")
    fnb = pd.concat([se_p, gu_p, dong_p], ignore_index=True)[["level", "series_id", "gu", "month"] + num]
    fnb.to_parquet(C.INTERIM / "fnb_monthly.parquet", index=False)
    json.dump({"categories": cats}, open(C.INTERIM / "fnb_categories.json", "w"), ensure_ascii=False)
    stats["fnb_dongs"] = D
    stats["fnb_categories"] = cats

json.dump(stats, open(C.INTERIM / "targets_stats.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k != "unmapped_categories"}, ensure_ascii=False, indent=1,
                 default=str))
print("매핑표에 없는 업태:", stats["unmapped_categories"])
log_run("00_targets", "대상 테이블 생성", stores=stats["stores_mapped_to_dong"], bulk=stats["bulk_closure_stores"])
