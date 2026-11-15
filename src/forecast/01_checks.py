"""개폐업 기본 점검: 행정동 결합률, 매핑 안 된 업태, 일괄 폐업, 폐업 신고 지연, 0 비율 -> reports/forecast/01_checks.md

신고 지연 점검은 데이터 품질 확인이라 2026년 기록도 본다(예측 결과나 오차는 보지 않는다).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C, log_run, mrange, read_csv, set_korean_font

set_korean_font()
R = C.REPORTS
stats = json.load(open(C.INTERIM / "targets_stats.json"))
st = pd.read_parquet(C.INTERIM / "stores_mapped.parquet")
fnb = pd.read_parquet(C.INTERIM / "fnb_monthly.parquet")
se = fnb[fnb.level == "seoul"].set_index("month")
dong = fnb[fnb.level == "dong"]
L = []
w = L.append

w("# 개폐업 기본 점검\n")
w("> `src/forecast/01_checks.py` 결과. 일반·휴게음식점 합침, 매핑표 `include == 1` 업종만.\n")

# ---------- 1. 행정동 결합 ----------
miss = pd.read_csv(C.INTERIM / "dong_unmapped_by_year.csv", encoding="utf-8-sig", index_col=0)
w("## 1. 행정동에 붙인 비율\n")
w(f"- 포함 업종 점포 {stats['stores_include']:,}개 중 **{stats['stores_mapped_to_dong']:,}개"
  f"({stats['stores_mapped_to_dong'] / stats['stores_include']:.1%})**를 좌표로 행정동에 붙였다.")
w(f"- 좌표를 못 쓰는 점포 {stats['coord_unusable']:,}개 (결측·범위 밖). 나머지 차이는 좌표가 서울 경계 밖인 점포.")
w("- 개업 연도별로 못 붙인 비율 (2018년 이후):\n")
w("| 개업 연도 | 점포 수 | 못 붙인 비율 |\n|---|---:|---:|")
for y, r in miss.loc[2018:].iterrows():
    w(f"| {y} | {int(r['size']):,} | {r['mean']:.1%} |")
w("\n못 붙인 점포는 동·구·서울 집계에서 모두 빠진다(서울 = 동 합계). 비율이 연도마다 비슷하면 추세 왜곡은 작다.\n")

# ---------- 2. 매핑 안 된 업태 ----------
unm = read_csv(C.UNMAPPED_CSV)
w("## 2. 매핑표에 없는 업태\n")
w(f"- {len(unm)}개 업태, {int(unm.n_stores.sum())}개 점포. 추측해서 넣지 않고 **제외**했다. 목록: `reports/unmapped_categories.csv`\n")
w("| 출처 | 업태 | 점포 수 |\n|---|---|---:|")
for _, r in unm.sort_values("n_stores", ascending=False).iterrows():
    w(f"| {r.source_type} | {r.business_type} | {r.n_stores} |")
w("")

# ---------- 3. 일괄 폐업 ----------
b = st[st.bulk_closure]
bev = b.groupby(["dong_code8", "gu", "close_date"]).size().rename("n").reset_index()
bev["year"] = bev.close_date.dt.year
dl = read_csv(C.DONG_LIST_CSV, dtype={"dong_code8": str}).set_index("dong_code8").dong_name
w("## 3. 일괄 폐업 (`bulk_closure`)\n")
w(f"- 규칙: 같은 동·같은 폐업일 건수 ≥ 그 동 일평균 × {C.BULK_MULT}, 그리고 ≥ {C.BULK_MIN}건 (2015년 이후 폐업 기준).")
w(f"- 해당: {len(bev):,}건(동·날짜), 점포 {len(b):,}개.\n")
yr = bev.groupby("year").agg(events=("n", "size"), stores=("n", "sum"))
allc = st[st.close_date.notna()].groupby(st.close_date.dt.year).size()
w("| 연도 | 일괄 폐업 건 | 점포 | 그해 전체 폐업 중 비율 |\n|---|---:|---:|---:|")
for y, r in yr.loc[2018:].iterrows():
    w(f"| {y} | {r.events} | {r.stores:,} | {r.stores / allc.get(y, np.nan):.1%} |")
w("\n가장 큰 일괄 폐업 10건:\n")
w("| 날짜 | 구 | 동 | 점포 |\n|---|---|---|---:|")
for _, r in bev[bev.year >= 2022].nlargest(10, "n").iterrows():
    w(f"| {r.close_date.date()} | {r.gu} | {dl.get(r.dong_code8, r.dong_code8)} | {r.n} |")
cm = se.loc["2022-01":"2025-12"]
moy = b[b.close_date >= "2022-01-01"].groupby(b.close_date.dt.month).size()
top3 = ", ".join(f"{m}월 {n / moy.sum():.0%}" for m, n in moy.nlargest(3).items())
w(f"\n- 2022~2025 서울 월 폐업: 유지(`v_keep`) 평균 {cm.C_v_keep.mean():.0f}, 제외(`v_drop`) 평균 {cm.C_v_drop.mean():.0f}.")
w(f"- 2022년 이후 일괄 폐업 점포가 많은 달: {top3}. 2024년부터 일괄 폐업이 두 배 넘게 늘어 폐업 추세를 왜곡할 수 있다.")
w("- 두 패널 모두 모델을 돌린다. 기본 보고는 `v_drop`. 영업 점포 수 S는 두 패널 공통(등록 상태 그대로).\n")

# ---------- 4. 폐업 신고 지연 ----------
w("## 4. 폐업 신고 지연\n")
c = st[st.close_date.notna()].copy()
c["lm"] = pd.to_datetime(c.last_modified_raw, errors="coerce")
c["lag"] = (c.lm - c.close_date).dt.days
mat = c[(c.close_date >= "2022-01-01") & (c.close_date <= "2024-12-31")]
neg = (mat.lag < 0).mean()
pos = mat.lag[mat.lag >= 0]
w("### 4.1 폐업일과 최종수정일 차이 (2022~2024 폐업)\n")
w(f"- 최종수정일이 폐업일보다 앞선 기록은 {neg:.1%}로 적다. 대부분 폐업 처리일에 기록이 수정됐다고 보고, "
  "차이가 0 이상인 기록으로 지연을 잰다. (2021년 이전 폐업은 수정일이 폐업일보다 앞선 기록이 많아 보지 않는다.)")
qs = pos.quantile([.5, .75, .9, .95]).round(0)
w(f"- 중앙값 {qs[.5]:.0f}일, 75% {qs[.75]:.0f}일, 90% {qs[.9]:.0f}일, 95% {qs[.95]:.0f}일.")
w(f"- 차이가 30일 이내: {(pos <= 30).mean():.0%}, 90일 이내: {(pos <= 90).mean():.0%}, 180일 이내: {(pos <= 180).mean():.0%}.\n")

w("### 4.2 최근 달 폐업 수를 전년 같은 달과 비교\n")
w("신고가 늦게 들어온다면 자료 끝 쪽 달의 폐업이 개업보다 더 크게 낮아 보인다. 개업 대비 폐업 비율(C/O)을 전년 같은 달과 비교한다.\n")
w("| 달 | 개업 | 폐업(v_drop) | C/O | 전년 같은 달 C/O |\n|---|---:|---:|---:|---:|")
for m in mrange("2025-07", "2026-10"):
    py = f"{int(m[:4]) - 1}{m[4:]}"
    r, q = se.loc[m], se.loc[py]
    w(f"| {m} | {r.O:,.0f} | {r.C_v_drop:,.0f} | {r.C_v_drop / r.O:.2f} | {q.C_v_drop / q.O:.2f} |")
ratio_recent = (se.loc["2026-01":"2026-09"].C_v_drop.sum() / se.loc["2026-01":"2026-09"].O.sum())
ratio_prev = (se.loc["2025-01":"2025-09"].C_v_drop.sum() / se.loc["2025-01":"2025-09"].O.sum())
w(f"\n- 2026-01~09 C/O {ratio_recent:.2f}, 2025-01~09 {ratio_prev:.2f}.")
w("- 2026-10은 부분 월(자료 받은 날까지)이라 쓰지 않는다.")
w("- 2026-05는 개업·폐업이 함께 낮다(전년 같은 달보다 각각 약 30%·35% 적음). 폐업만 낮은 것이 아니라 신고 지연의 모양은 아니다. 원인은 확인하지 못했다.")
w("- **판단**: 폐업 기록의 99%가 폐업일 30일 안에 들어오고, 최근 달의 C/O도 전년과 비슷해 **신고 지연 영향은 작다**. "
  "개발 fold 실제값(~2025-12)은 영향이 없다고 본다. 최종 시험(2026-01~09)도 지금 자료로 평가할 수 있지만, "
  "마지막 달(2026-09)은 소수 누락이 있을 수 있어 결과에 함께 적는다.\n")

# ---------- 5. 0 비율과 적용 범위 ----------
d = dong[(dong.month >= "2022-01") & (dong.month <= "2023-12")]
g = d.groupby("series_id")
z = pd.DataFrame({"zO": g.O.apply(lambda v: (v == 0).mean()), "zC": g.C_v_drop.apply(lambda v: (v == 0).mean()),
                  "oc": g.apply(lambda x: (x.O + x.C_v_drop).mean())})
n_cro = int(((z.zO >= C.CROSTON_ZERO_SHARE) | (z.zC >= C.CROSTON_ZERO_SHARE)).sum())
n_busy = int((z.oc >= C.BUSY_DONG_MIN).sum())
w("## 5. 동별 0 비율과 모델 적용 범위 (F2024 학습 구간 2022~2023 기준)\n")
w(f"- 동 {len(z)}개. 월 개업 중앙값 {d.O.median():.0f}건, 폐업 중앙값 {d.C_v_drop.median():.0f}건.")
w(f"- 개업 또는 폐업이 0인 달이 {C.CROSTON_ZERO_SHARE:.0%} 이상인 동(Croston/TSB 적용): **{n_cro}개**")
w(f"- 월평균 개업+폐업 ≥ {C.BUSY_DONG_MIN}건인 동(SARIMA 적용): **{n_busy}개**")
w(f"- 0인 달 비율 분포: 개업 평균 {z.zO.mean():.1%}, 폐업 평균 {z.zC.mean():.1%}\n")

# ---------- 그림 ----------
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
x = se.loc["2018-01":"2026-09"]
ax[0].plot(x.index, x.O, label="개업", color="#2a6f97")
ax[0].plot(x.index, x.C_v_keep, label="폐업 (일괄 폐업 포함, v_keep)", color="#c0504d", alpha=.5)
ax[0].plot(x.index, x.C_v_drop, label="폐업 (일괄 폐업 제외, v_drop)", color="#c0504d")
ax[0].axvspan("2022-01", "2022-04", color="grey", alpha=.15)
ax[0].legend(fontsize=8, ncol=3)
ax[0].set_title("서울 음식점 월별 개업·폐업 (행정동에 붙은 점포)")
ax[1].plot(x.index, x.S / 1000, color="#444")
ax[1].set_ylabel("영업 점포 (천)")
ticks = [m for m in x.index if m.endswith("-01")]
ax[1].set_xticks(ticks)
ax[1].set_xticklabels([t[:4] for t in ticks])
for a in ax:
    a.axvline("2022-01", color="k", lw=.5, ls=":")
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(R / "01_checks_seoul_monthly.png", dpi=150)
w("![서울 월별 개폐업](01_checks_seoul_monthly.png)\n")

z.to_csv(R / "01_checks_dong_zero_share.csv", encoding="utf-8-sig")
(R / "01_checks.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
log_run("01_checks", "개폐업 기본 점검", croston_dongs=n_cro, busy_dongs=n_busy, bulk_stores=len(b))
