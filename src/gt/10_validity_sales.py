"""G.T 타당성 검증 3: 모델 입력이 아닌 상업 인기 근거 (피드백 2번)
자료: 서울시 상권분석서비스 추정매출-행정동 (OA-22175), 분기. 카드 매출 기반 추정치로 생활이동·검색·인허가·지하철과 출처가 다르다
  - 음식업 매출 건수 (서비스 업종 CS100*), 전체 업종 매출 건수
비교:
  A. 수준: 2025년 1·2분기 평균 vs 생활이동 2025-01~06 평균 (동별 순위 상관, 핫스팟 겹침)
  B. 변화: 2024 -> 2025 (1·2분기) 서울 대비 비중 변화의 동별 순위 상관 (전체 동)
     상권분석서비스는 2024년부터 공간 기준이 바뀌어 2023 -> 2024 변화는 참고로만 본다
  C. 2024-08~10 생활이동 이상이 매출에도 있나: 서울 합계 분기별 추이 (3분기 급증 여부)
출력: reports/gt_midterm/tables/v3_*.csv, fig/V3_*.png
"""
import glob, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from common import log_run, MERGE, LP_RENAME, set_korean_font

SRC = sys.argv[1] if len(sys.argv) > 1 else str(C.ROOT / "data/raw/sales")
to_unit = lambda c: MERGE.get(LP_RENAME.get(c, c), LP_RENAME.get(c, c))
frames = []
for f in sorted(glob.glob(SRC + "/*.csv")):
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(f, encoding=enc, usecols=["기준_년분기_코드", "행정동_코드", "서비스_업종_코드", "당월_매출_건수", "당월_매출_금액"], dtype={"행정동_코드": str}); break
        except Exception: d = None
    frames.append(d)
S = pd.concat(frames, ignore_index=True)
S["unit"] = S["행정동_코드"].str[:8].map(to_unit)
S["food"] = S["서비스_업종_코드"].str.startswith("CS1")
q = S.groupby(["unit", "기준_년분기_코드"]).agg(all_cnt=("당월_매출_건수", "sum")).join(
    S[S.food].groupby(["unit", "기준_년분기_코드"]).agg(food_cnt=("당월_매출_건수", "sum"), food_amt=("당월_매출_금액", "sum"))).reset_index()
q = q.rename(columns={"기준_년분기_코드": "yq"})

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
T["yq"] = T.week_end.dt.year * 10 + T.week_end.dt.quarter
gq = T.groupby(["unit", "yq"]).y.mean().rename("mob").reset_index()
M = gq.merge(q, on=["unit", "yq"], how="inner")
VARS = {"food_cnt": "음식업 매출 건수", "food_amt": "음식업 매출 금액", "all_cnt": "전체 업종 매출 건수"}

# A. 수준
L = M[M.yq.isin([20251, 20252])].groupby("unit")[["mob"] + list(VARS)].mean()
rows = []
n_top = lambda n: int(round(n * C.HOT_TOP))
for v, lab in VARS.items():
    d = L[["mob", v]].dropna(); d = d[d[v] > 0]
    r, p = stats.spearmanr(d.mob, d[v]); k = n_top(len(d))
    ov = len(set(d.mob.nlargest(k).index) & set(d[v].nlargest(k).index)) / k
    rows.append(("수준 (2025 1·2분기)", lab, len(d), r, p, ov))

# B. 변화 (서울 대비 비중)
def share_chg(y0, y1):
    a = M[M.yq.isin([y0 * 10 + 1, y0 * 10 + 2])].groupby("unit")[["mob"] + list(VARS)].mean()
    b = M[M.yq.isin([y1 * 10 + 1, y1 * 10 + 2])].groupby("unit")[["mob"] + list(VARS)].mean()
    a, b = a / a.sum(), b / b.sum()
    return (b / a - 1).replace([np.inf, -np.inf], np.nan)
CH = {}
for y0, y1, tag in ((2024, 2025, "변화 2024→2025 (주)"), (2023, 2024, "변화 2023→2024 (참고, 공간 기준 변경)")):
    ch = share_chg(y0, y1); CH[tag] = ch
    for v, lab in VARS.items():
        d = ch[["mob", v]].dropna()
        r, p = stats.spearmanr(d.mob, d[v]); k = n_top(len(d))
        ov = len(set(d.mob.nlargest(k).index) & set(d[v].nlargest(k).index)) / k
        rows.append((tag, lab, len(d), r, p, ov))
R = pd.DataFrame(rows, columns=["비교", "자료", "동 수", "스피어만 상관", "p", "상위 10% 겹침"])
R.round(4).to_csv(C.TABLES / "v3_sales.csv", index=False, encoding="utf-8-sig")

# C. 서울 합계 분기 추이
Sq = M.groupby("yq")[["mob", "food_cnt", "all_cnt"]].sum()
Sq = Sq / Sq.loc[[20241, 20242]].mean()
Sq.round(3).to_csv(C.TABLES / "v3_seoul_quarterly.csv", encoding="utf-8-sig")

set_korean_font()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5})
f, axs = plt.subplots(1, 3, figsize=(14, 4))
d = L[["mob", "food_cnt"]].dropna(); d = d[d.food_cnt > 0]
axs[0].scatter(d.food_cnt, d.mob, s=9, color="#6E86A8", alpha=.7, lw=0); axs[0].set_xscale("log"); axs[0].set_yscale("log")
r0 = R[(R["비교"].str.startswith("수준")) & (R["자료"] == "음식업 매출 건수")].iloc[0]
axs[0].set_title(f"수준 (2025 상반기): 순위 상관 {r0['스피어만 상관']:.2f}", loc="left"); axs[0].set_xlabel("음식업 매출 건수 (분기)"); axs[0].set_ylabel("생활이동 (일평균)")
ch = CH["변화 2024→2025 (주)"][["mob", "food_cnt"]].dropna(); ch = ch[(ch.abs() < 1.5).all(axis=1)]
axs[1].scatter(ch.food_cnt * 100, ch.mob * 100, s=9, color="#6E86A8", alpha=.7, lw=0); axs[1].axhline(0, color="#8A8F98", lw=.6); axs[1].axvline(0, color="#8A8F98", lw=.6)
r1 = R[(R["비교"] == "변화 2024→2025 (주)") & (R["자료"] == "음식업 매출 건수")].iloc[0]
axs[1].set_title(f"변화 2024→2025 (서울 대비 비중): 순위 상관 {r1['스피어만 상관']:.2f}", loc="left"); axs[1].set_xlabel("음식업 매출 건수 비중 변화 (%)"); axs[1].set_ylabel("생활이동 비중 변화 (%)")
lab_q = [f"{str(i)[:4]}.{str(i)[4]}Q" for i in Sq.index]
axs[2].plot(lab_q, Sq.mob, marker="o", color="#D4462A", label="생활이동"); axs[2].plot(lab_q, Sq.food_cnt, marker="o", color="#2F5FB3", label="음식업 매출 건수")
axs[2].plot(lab_q, Sq.all_cnt, marker="o", color="#8A8F98", label="전체 업종 매출 건수")
axs[2].set_title("서울 합계 (2024 상반기 = 1): 2024-3분기 급증은 생활이동에만", loc="left"); axs[2].legend(frameon=False, fontsize=8); axs[2].tick_params(axis="x", rotation=45)
f.suptitle("G.T 타당성 3: 모델 입력이 아닌 카드 매출(상권분석서비스 추정매출-행정동)과 비교", x=0.01, ha="left", fontsize=12)
f.tight_layout(); f.savefig(C.FIG / "V3_sales.png", bbox_inches="tight"); plt.close(f)
log_run("10_validity_sales", f"level={r0['스피어만 상관']:.2f}; change={r1['스피어만 상관']:.2f}")
print(R.round(3).to_string(index=False)); print(Sq.round(2).to_string())
