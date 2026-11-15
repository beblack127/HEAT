"""정답 주 단위 표 (명세 §3).
입력: data/raw/mobility/mob_YYYYMM.parquet (reduce_mobility.py 결과)
출력: data/interim/gt/units.csv, target_weekly.parquet, reports/gt_midterm/00_data_audit.md
"""
import glob
import numpy as np, pandas as pd
import config as C
from common import log_run, unit_map

mob = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RAW_MOB / "mob_*.parquet")))], ignore_index=True)
raw_codes = sorted(mob.dong_code8.unique())
U = unit_map()                                   # 427동 -> 분석 단위
mob["unit"] = mob.dong_code8.map(U.set_index("src_code").unit) if False else mob.dong_code8.map(dict(zip(U.src_code, U.unit)))
unmatched = sorted(set(mob.loc[mob.unit.isna(), "dong_code8"]))

# 일 값: 목적별 -> 정의별 합. 없는 목적은 0
def daily(purposes, col):
    d = mob[mob.purpose.isin(purposes)].groupby(["dong_code8", "date"])[col].sum()
    return d
days = mob.groupby(["dong_code8", "date"]).size().rename("n").reset_index()[["dong_code8", "date"]]
D = days.set_index(["dong_code8", "date"])
D["y"] = daily(C.TARGET_PURPOSES, "y2030"); D["y_p45"] = daily(C.SENS_PURPOSES, "y2030"); D["y_allage"] = daily(C.TARGET_PURPOSES, "yall")
D = D.fillna(0.0).reset_index()
D = D[D.date >= C.DATA_START]
D["week_end"] = D.date + pd.to_timedelta((6 - D.date.dt.weekday) % 7, unit="D")

# 서울 합계 (원자료의 모든 서울 동)
seoul_day = D.groupby("date")[["y", "y_p45", "y_allage"]].sum().reset_index()
seoul_day["week_end"] = seoul_day.date + pd.to_timedelta((6 - seoul_day.date.dt.weekday) % 7, unit="D")
seoul_w = seoul_day.groupby("week_end").agg(seoul_y=("y", "mean"), seoul_y_p45=("y_p45", "mean"), seoul_y_allage=("y_allage", "mean"), n_days_seoul=("date", "nunique"))

# 분석 단위로 합친 뒤 주 일평균
D["unit"] = D.dong_code8.map(dict(zip(U.src_code, U.unit)))
D = D.dropna(subset=["unit"])
du = D.groupby(["unit", "date", "week_end"], as_index=False)[["y", "y_p45", "y_allage"]].sum()
W = du.groupby(["unit", "week_end"]).agg(y=("y", "mean"), y_p45=("y_p45", "mean"), y_allage=("y_allage", "mean"), n_days=("date", "nunique")).reset_index()
W.loc[W.n_days < C.MIN_DAYS, ["y", "y_p45", "y_allage"]] = np.nan
W = W.merge(seoul_w, on="week_end", how="left")
W["share"] = W.y / W.seoul_y
W["share_p45"] = W.y_p45 / W.seoul_y_p45
W["share_allage"] = W.y_allage / W.seoul_y_allage
# 마지막 주가 부분 주면 뺀다
last_full = W.loc[W.n_days_seoul == 7, "week_end"].max()
W = W[W.week_end <= last_full]

# 품질 표시 (처리하지 않음)
W = W.sort_values(["unit", "week_end"])
W["y_lag52"] = W.groupby("unit").y.shift(52)
r = np.log1p(W.y) - np.log1p(W.y_lag52)
mad = W.assign(r=r).groupby("unit").r.transform(lambda v: (v - v.median()).abs().median())
W["flag_spike"] = ((r - r.groupby(W.unit).transform("median")).abs() > 5 * mad.replace(0, np.nan)).astype(int)
wk = W.week_end
W["flag_quality_period"] = (((wk >= "2024-08-01") & (wk <= "2024-10-31")) | (wk >= "2025-09-01")).astype(int)
W.drop(columns=["y_lag52"]).to_parquet(C.INTERIM / "target_weekly.parquet", index=False)

# 점검 기록
n_units = W.unit.nunique()
lines = ["# 생활이동 자료 점검", "",
         f"- 출처: 수도권 생활이동 (성 연령별, 도착지 기준)-내국인 (OA-22298), 월 파일 {mob.date.dt.to_period('M').nunique()}개",
         f"- 기간: {mob.date.min().date()} ~ {mob.date.max().date()}, 주 단위 {W.week_end.min().date()} ~ {W.week_end.max().date()} ({W.week_end.nunique()}주)",
         f"- 원자료 서울 도착 동 코드: {len(raw_codes)}개, 분석 단위: {n_units}개 (427동 기준, 아래 대응 반영)",
         f"- 정답: 20대+30대 남녀, 목적 {C.TARGET_PURPOSES}, 하루 값 = 모든 시간대 합, 주 값 = 유효 일 평균",
         f"- 대응 안 된 원자료 코드: {unmatched if unmatched else '없음'}",
         "", "## 목적별 비중 (서울 전체, 20~39세, 기간 합)", "",
         "| 목적 | 비중 |", "|---|---|"]
pc = mob.groupby("purpose").y2030.sum(); pc = pc / pc.sum()
names = {1: "출근", 2: "등교", 3: "귀가", 4: "쇼핑", 5: "관광", 6: "병원", 7: "기타"}
lines += [f"| {k} {names.get(k, '')} | {v:.1%} |" for k, v in pc.items()]
lines += ["", "목적 4+5+7 중 대부분이 7(기타)이다. 그래서 4+5만 쓴 민감도 정의를 함께 본다.", "",
          "## 동 코드 대응", "", "| 원자료 코드 | 분석 단위 | 사유 |", "|---|---|---|"]
lines += [f"| {r.src_code} | {r.unit} | {r.reason} |" for r in U[U.reason != ""].itertuples()]
lines += ["", f"- 품질 구간 표시: 2024-08~10, 2025-09 이후 (처리하지 않음)",
          f"- 급등·급락 표시(작년 같은 주 대비 MAD 5배): {int(W.flag_spike.sum())}개 동·주"]
(C.REPORT / "00_data_audit.md").write_text("\n".join(lines), encoding="utf-8")
log_run("00_target", f"units={n_units}; weeks={W.week_end.nunique()}")
print("\n".join(lines[:6]))
