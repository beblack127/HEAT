"""입력 주 단위 표 (명세 §4). 모든 입력은 고정 시차라서 예측 13주 동안 예측 시점에 이미 아는 값이다.
- 월 단위(검색, 음식점): 주 t가 속한 달 - LAG_MONTHLY_X(4)개월의 값
- 주 단위(지하철, 생활인구): t - LAG_WEEKLY_X(13)주의 값
- 공휴일: 시차 없음 / 작년 같은 주(y_lag52): 52주 전 정답
출력: data/interim/gt/features_weekly.parquet, reports/gt_midterm/00_unmatched_stations.csv
"""
import glob
import numpy as np, pandas as pd, holidays
import config as C
from common import log_run, MERGE, LP_RENAME, LP_SPLIT_PARENT, dong_list

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
weeks = pd.DatetimeIndex(sorted(T.week_end.unique()))
weeks = weeks.append(pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=C.H, freq="W-SUN"))   # 마지막 주 뒤 13주 (실시간 예측용)
units = sorted(T.unit.unique())
to_unit = lambda c: MERGE.get(c, c)
grid = pd.MultiIndex.from_product([units, weeks], names=["unit", "week_end"]).to_frame(index=False)
grid["month_ref"] = (grid.week_end.dt.to_period("M") - C.LAG_MONTHLY_X).astype(str)

# ---------- 검색 (월) ----------
S = pd.read_csv(C.SEARCH_CSV, encoding="utf-8-sig", dtype={"dong_code8": str})
S = S[S.tier == "전체"].assign(unit=lambda d: d.dong_code8.map(to_unit))
S = S.groupby(["unit", "month"], as_index=False)[["seoul_share", "index"]].sum(min_count=1)
grid = grid.merge(S.rename(columns={"month": "month_ref", "seoul_share": "search_share_raw", "index": "search_index_raw"}), on=["unit", "month_ref"], how="left")
grid["search_share"] = np.log(grid.search_share_raw * 1e4)
grid["search_index"] = np.log1p(grid.search_index_raw)

# ---------- 음식점 (월): 영업 점포 12개월 변화율 ----------
F = pd.read_csv(C.FNB_CSV, encoding="utf-8-sig", dtype={"code": str})
F = F[F.level == "dong"].assign(unit=lambda d: d.code.map(to_unit)).groupby(["unit", "month"], as_index=False)["영업점포"].sum()
F = F.sort_values(["unit", "month"])
F["fnb_stock_chg12"] = F.groupby("unit")["영업점포"].transform(lambda v: v / v.shift(12) - 1)
grid = grid.merge(F[["unit", "month", "fnb_stock_chg12"]].rename(columns={"month": "month_ref"}), on=["unit", "month_ref"], how="left")

# ---------- 지하철 (주): 동 하차 4주 평균, 13주 전 ----------
sub = pd.read_parquet(C.RAW_SUB / "subway_daily.parquet")
lk = pd.read_csv(C.STA_LINK, encoding="utf-8-sig", dtype={"dong_code8": str})
lk["station"] = lk.station.replace({"서울역": "서울"})   # 이름표의 '서울역' = 지하철 자료의 '서울'
lk = lk.groupby(["station", "dong_code8"], as_index=False).share_1000.max()
land = dong_list().set_index("dong_code8").land_km2
lk["w"] = lk.share_1000 * lk.dong_code8.map(land)
lk = lk[lk.w > 0]
lk["w"] = lk.w / lk.groupby("station").w.transform("sum")
miss = sorted(set(lk.station) - set(sub.station))
pd.DataFrame({"station": miss, "reason": "지하철 자료(OA-12914)에 없음"}).to_csv(C.REPORT / "00_unmatched_stations.csv", index=False, encoding="utf-8-sig")
sub["week_end"] = sub.date + pd.to_timedelta((6 - sub.date.dt.weekday) % 7, unit="D")
sw = sub.groupby(["station", "week_end"], as_index=False).alight.mean()
dw = sw.merge(lk[["station", "dong_code8", "w"]], on="station")
dw["v"] = dw.alight * dw.w
dw = dw.assign(unit=dw.dong_code8.map(to_unit)).groupby(["unit", "week_end"], as_index=False).v.sum()
dw = dw.sort_values(["unit", "week_end"])
dw["subway_exit_4w"] = np.log1p(dw.groupby("unit").v.transform(lambda v: v.rolling(4, min_periods=4).mean()))
dw["subway_exit_4w"] = dw.groupby("unit").subway_exit_4w.shift(C.LAG_WEEKLY_X)
grid = grid.merge(dw[["unit", "week_end", "subway_exit_4w"]], on=["unit", "week_end"], how="left")

# ---------- 생활인구 (주): 주간 평균, 13주 전 ----------
lp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RAW_LP / "lp_*.parquet")))], ignore_index=True)
lp["dong_code8"] = lp.dong_code8.replace(LP_RENAME)
# 나뉘기 전 코드는 나뉜 뒤 첫 달의 자식 비율로 나눈다 (없으면 똑같이)
for parent, kids in LP_SPLIT_PARENT.items():
    if parent not in set(lp.dong_code8): continue
    after = lp[lp.dong_code8.isin(kids)]
    if len(after):
        first = after.date.min(); r = after[after.date == first].set_index("dong_code8").lp_all
        ratio = (r / r.sum()).reindex(kids).fillna(1 / len(kids))
    else:
        ratio = pd.Series(1 / len(kids), index=kids)
    par = lp[lp.dong_code8 == parent]
    lp = pd.concat([lp[lp.dong_code8 != parent]] + [par.assign(dong_code8=k, lp_all=par.lp_all * ratio[k], lp_2039=par.lp_2039 * ratio[k]) for k in kids])
lp["unit"] = lp.dong_code8.map(to_unit)
lp["week_end"] = lp.date + pd.to_timedelta((6 - lp.date.dt.weekday) % 7, unit="D")
lpw = lp.groupby(["unit", "date", "week_end"], as_index=False).lp_all.sum().groupby(["unit", "week_end"], as_index=False).lp_all.mean()
lpw = lpw.sort_values(["unit", "week_end"])
lpw["living_pop"] = np.log1p(lpw.groupby("unit").lp_all.shift(C.LAG_WEEKLY_X))
grid = grid.merge(lpw[["unit", "week_end", "living_pop"]], on=["unit", "week_end"], how="left")

# ---------- 공휴일, 작년 같은 주 ----------
kr = holidays.KR(years=range(2022, 2028))
hd = pd.Series(1, index=pd.DatetimeIndex(list(kr.keys()))).sort_index()
grid["holiday_days"] = [int(((hd.index > w - pd.Timedelta(days=7)) & (hd.index <= w)).sum()) for w in grid.week_end]
T = T.sort_values(["unit", "week_end"])
T["y_lag52"] = np.log1p(T.groupby("unit").y.shift(C.SEASON))
grid = grid.merge(T[["unit", "week_end", "y_lag52"]], on=["unit", "week_end"], how="left")

grid.drop(columns=["month_ref"]).to_parquet(C.INTERIM / "features_weekly.parquet", index=False)

# ---------- 누수 시험 ----------
msgs = []
for o in pd.to_datetime(C.ORIGINS):
    fut = weeks[(weeks > o) & (weeks <= o + pd.Timedelta(weeks=C.H))]
    mref_end = (fut.to_period("M") - C.LAG_MONTHLY_X).to_timestamp(how="end")
    assert (mref_end <= o).all(), f"월 입력 누수 {o}"
    assert ((fut - pd.Timedelta(weeks=C.LAG_WEEKLY_X)) <= o).all(), f"주 입력 누수 {o}"
    assert ((fut - pd.Timedelta(weeks=C.SEASON)) <= o).all(), f"작년 같은 주 누수 {o}"
    msgs.append(f"{o.date()}: 예측 {len(fut)}주, 월 입력 최신 {mref_end.max().date()}, 주 입력 최신 {(fut - pd.Timedelta(weeks=C.LAG_WEEKLY_X)).max().date()} (모두 시점 이전)")
(C.REPORT / "00_leakage_test.md").write_text("# 누수 시험\n\n" + "\n".join(f"- {m}" for m in msgs), encoding="utf-8")
cov = grid.notna().mean().round(3).to_dict()
log_run("01_features", f"miss_stations={len(miss)}")
print("\n".join(msgs)); print("결측 아닌 비율:", {k: v for k, v in cov.items() if k not in ("unit", "week_end")}); print("못 맞춘 역:", miss)
