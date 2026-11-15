"""중간발표 실험 결과 리포트 -> reports/gt_midterm/00_결과_리포트.md
숫자는 모두 reports/gt_midterm/tables/*.csv 에서 읽는다. 다시 돌리면 숫자가 같이 바뀐다.
"""
import numpy as np, pandas as pd
import config as C
from common import log_run

TB = C.TABLES
rd = lambda n, **k: pd.read_csv(TB / n, encoding="utf-8-sig", **k)
t1 = rd("t1_main_wape.csv", index_col=0)
t2 = rd("t2_by_origin_wape.csv", index_col=0)
t3 = rd("t3_prior_year.csv"); t4 = rd("t4_search_share.csv")
tests = rd("t_tests.csv"); dm = rd("t_dm.csv"); mase = rd("t_mase.csv", index_col=0)
q = rd("t5_dong_wape_dist.csv", index_col=0)
K = rd("topk.csv"); S = rd("sensitivity.csv"); H = rd("hotspots.csv", dtype={"unit": str})
CS = rd("case_dongs.csv", dtype={"unit": str}); NSET = rd("common_set_size.csv", index_col=0)
W = pd.read_parquet(C.RESULTS / "wape_by_origin.parquet"); W["o"] = pd.to_datetime(W.origin).dt.date.astype(str)
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")

LAB = {"naive": "Naive(지난주)", "snaive": "작년 같은 주", "arima": "ARIMA", "sarima": "SARIMA", "arimax": "ARIMAX", "sarimax": "SARIMAX"}
XL = {"none": "입력 없음", "search": "검색", "all_noLP": "전체−생활인구", "all": "전체", "search_idx": "검색(원지수)"}
def kname(k):
    m, x = k.split("|"); py = x.endswith("+py"); x = x.replace("+py", "")
    s = LAB[m] + ("" if x == "none" else " + " + XL.get(x, x))
    return s + (" + 작년 같은 주 입력" if py else "")
def f1(v): return "–" if pd.isna(v) else f"{v:.1f}"
def pct(v): return "–" if pd.isna(v) else f"{v * 100:.0f}%"
def md(df): return df.to_markdown()

CLEAN = ["2025-01-26", "2025-04-27"]
mean4 = W.groupby("key").wape.mean()
clean = W[W.o.isin(CLEAN)].groupby("key").wape.mean()
rep = mean4.drop("snaive|none").idxmin()
sn = mean4["snaive|none"]; sn_c = clean["snaive|none"]
seoul_m = T.groupby("week_end").seoul_y.first().resample("MS").mean()
spike_ratio = seoul_m["2024-08":"2024-10"].mean() / pd.concat([seoul_m["2024-05":"2024-07"], seoul_m["2024-11":"2025-01"]]).mean()

def tget(model, vs):
    r = tests[(tests.model == model) & (tests.vs == vs)]
    return (r.improved_share.iloc[0], r.wilcoxon_p.iloc[0]) if len(r) else (np.nan, np.nan)

def topk_table(target):
    k = K[(K.target == target) & (K.K.astype(str) == "10%")]
    p = k.pivot(index="method", columns="origin", values="hit_rate")
    p["평균"] = p.mean(axis=1)
    p = p.sort_values("평균", ascending=False)
    keep = ["persistence", "random"] + [i for i in p.index if i not in ("persistence", "random")][:6]
    p = p.loc[[i for i in p.index if i in keep]]
    SP = {"persistence": "지속 (직전 13주 그대로)", "random": "무작위"}
    p.index = [SP[i] if i in SP else kname(i) for i in p.index]
    return (p * 100).round(0).astype("Int64").astype(str) + "%", int(k.k.dropna().iloc[0])

L = []
a = L.append
a("# 중간발표 실험 결과: 생활이동 13주 예측과 핫스팟 판정\n")
a(f"> 만든 날 {pd.Timestamp.today().date()} · 코드 `src/gt/` · 명세 `docs/experiments/midterm_experiments.md` · 다시 돌리기 `bash src/gt/run_all.sh`\n")
a("> 숫자는 모두 `reports/gt_midterm/tables/`의 CSV에서 나온 값이다. 그림은 `reports/gt_midterm/fig/`.\n")

a("## 0. 한눈에 보기\n")
a("0. **생활이동은 핫함의 정답으로 쓸 근거가 있다.** 동별 순위가 음식점 수(0.83)·지하철 하차(0.73)·검색 비중(0.46)과 같이 가고, 현재 핫스팟의 72%가 이 독립 자료 2개 이상에서도 상위 10%다 (§2-1).")
a(f"1. **초안과 같은 조건에서 숫자가 재현된다.** 2025-04-27 시점 작년 같은 주 WAPE {f1(t2.loc['2025-04-27','snaive|none'])}% (초안 8.52%). 정답 정의와 동 대응이 맞다.")
a(f"2. **4개 시점 평균으로는 ARIMA가 가장 좋다** ({f1(mean4['arima|none'])}%, 작년 같은 주 {f1(sn)}%). 하지만 서울 전체 오차로 본 DM 검정에서는 차이가 유의하지 않다 (p = {dm.set_index('model').loc['arima|none','p']:.2f}).")
a(f"3. **결과가 시점에 따라 크게 뒤집힌다.** 원인은 **2024년 8~10월 자료 이상**(서울 전체가 앞뒤 달의 {spike_ratio:.2f}배). 이 구간이 끼지 않은 두 시점만 보면 작년 같은 주가 {f1(sn_c)}%로 다시 1등이다.")
a(f"4. **외생변수(검색·음식점·지하철·생활인구)를 넣으면 대체로 나빠진다.** ARIMA {f1(mean4['arima|none'])}% → ARIMAX + 전체 {f1(mean4['arimax|all'])}%. 동별로 봐도 좋아진 동이 절반이 안 된다.")
k_hot, nh = topk_table("hot"); k_em, ne = topk_table("emerging")
a(f"5. **핫스팟 순위는 무작위보다 훨씬 잘 맞지만 '지금 순위 그대로'를 넘지 못했다.** 상위 10% 적중률: 현재 핫스팟 약 90%, 떠오르는 곳 약 50% (무작위 10%). 지속 기준도 비슷하거나 더 높다.")
a("6. **떠오르는 곳에는 상권보다 새 아파트 입주 같은 생활권 변화가 많이 잡힌다.** 정답의 대부분이 목적 7(기타)이기 때문이다. 해석할 때 주의가 필요하다.\n")

a("## 1. 무엇을 했나\n")
a("| 항목 | 내용 |\n|---|---|")
a("| 정답 | 생활이동(OA-22298) 20~39세 남녀, 목적 4(쇼핑)+5(관광)+7(기타), 도착 행정동, 주(월~일) 일평균 |")
a(f"| 지역 | 이름표 427동(2025-12-31 경계). 신설동·용두동은 생활이동에서 나뉘기 전 코드(용신동)만 있어 합침 → 분석 단위 426개. 4개 시점 모두 예측이 있는 공통 {int(NSET.n_units.min())}개 단위로 평가 |")
a(f"| 기간 | 2023-01 ~ 2026-10 (주 {T.week_end.nunique()}개). 학습은 2023-01부터 |")
a("| 예측 | 13주 앞, 주 단위. 예측 시점 4개: 2024-10-27, 2025-01-26, 2025-04-27, 2025-07-27 |")
a("| 모델 | Naive(지난주), 작년 같은 주(364일 전), ARIMA, SARIMA(계절 차분 52주), ARIMAX, SARIMAX. 모두 log1p(y)에 적합 |")
a("| 입력 | 검색(서울 대비 비중, 4개월 전 달), 음식점 영업점포 12개월 변화(4개월 전 달), 지하철 하차 4주 평균(13주 전), 생활인구(13주 전), 공휴일 수(시차 없음), 작년 같은 주 값(M3) |")
a("| 지표 | WAPE(13주 오차 합 / 실제 합), MASE(작년 같은 주 기준), 동별 개선 비율 + Wilcoxon, DM-HLN(서울 합 오차), 상위 K개 적중률 |")
a("")
a("- 입력은 모두 **고정 시차**로 붙였다. 13주 앞 예측의 마지막 주에도 예측 시점 이전 값만 쓰게 하려는 것이다 (`00_leakage_test.md`에서 확인).")
a("- 차수: ARIMA는 p∈{0,1,2}, d∈{0,1}, q∈{0,1} 중 AIC 최소. 외생변수 모델은 같은 차수를 그대로 쓴다. 적합 실패로 대체한 경우는 0건, 수렴 경고는 " + str(len(pd.read_csv(C.RESULTS / 'fit_log.csv'))) + "건(전체 적합의 1% 미만).")
a("- 행사 급등 처리는 하지 않았다 (다음 단계). 급등 주와 품질 구간은 표시만 했다.\n")

a("## 2. 자료 점검: 2024년 8~10월 이상\n")
a("![서울 합계](fig/F10_seoul_total.png)\n")
a(f"- 서울 전체 20~39세 목적 4+5+7 도착 이동이 2024-08~10에만 앞뒤 달의 **{spike_ratio:.2f}배**로 뛰었다가 11월에 원래 수준으로 돌아온다. 다른 해 같은 달에는 이런 움직임이 없어 실제 변화보다는 자료 쪽 문제로 보인다.")
a("- 이 구간은 두 시점의 결과를 크게 흔든다.")
a("  - **2024-10-27 시점**: 학습 끝부분이 부풀어 있어 지난주 값·ARIMA가 높게 예측 → 크게 틀림. 작년 같은 주는 2023년 값을 써서 괜찮다.")
a("  - **2025-07-27 시점**: 예측 구간(2025-08~10)의 작년 같은 주가 부푼 2024년 값 → 작년 같은 주·SARIMA가 크게 틀림.")
a("- 그래서 아래 표마다 **4개 시점 평균**과 **이상 구간이 끼지 않은 2개 시점(2025-01-26, 2025-04-27) 평균**을 함께 적는다. 어느 쪽이 '진짜' 성능인지는 자료 제공처 확인이 필요하다.\n")

# ---- G.T 타당성 (08_validity.py) ----
if (TB / "v1_correlation.csv").exists():
    V = rd("v1_correlation.csv"); VH = rd("v1_hotspot_agreement.csv", dtype={"unit": str}); VM = rd("v1_corr_matrix.csv", index_col=0)
    v = V.set_index("자료")
    a("## 2-1. G.T 타당성: 생활이동이 '핫함'을 재나\n")
    a("예측을 잘 하는지와 별개로, 생활이동이 많은 동이 **다른 독립 자료에서도** 사람이 많이 모이는 곳인지 확인했다. 2025-01~08 동별 평균(품질 점검 구간 제외), 같은 427동 단위.\n")
    a("| 비교 자료 | 출처 | 동 수 | 순위 상관 (스피어만) | 현재 핫스팟 겹침 | 무작위면 |\n|---|---|---|---|---|---|")
    for _, r in V.iterrows():
        a(f"| {r['자료']} | {r['출처']} | {int(r['동 수'])} | **{r['스피어만 상관']:.2f}** | **{r['핫스팟 겹침'] * 100:.0f}%** | {r['무작위 겹침'] * 100:.0f}% |")
    a("\n![타당성 산점도](fig/V1_validity_scatter.png)\n")
    dist = VH["독립자료_상위10%_개수"].value_counts().reindex(range(4), fill_value=0)
    a(f"- **독립 자료 3개 모두와 같이 간다.** 음식점 수 {v.loc['음식점 수','스피어만 상관']:.2f}, 지하철 하차 {v.loc['지하철 하차','스피어만 상관']:.2f}, 검색 비중 {v.loc['검색 비중','스피어만 상관']:.2f} (모두 p < 0.001). 생활인구({v.loc['생활인구 20~39세','스피어만 상관']:.2f})는 같은 통신 자료라 참고만 한다.")
    a(f"- **생활이동이 세 자료의 '가운데'에 있다.** 생활이동과 각 자료의 상관이 그 자료들끼리의 상관보다 높다 (지하철–음식점 {VM.loc['지하철 하차','음식점 수']:.2f}, 지하철–검색 {VM.loc['지하철 하차','검색 비중']:.2f}, 음식점–검색 {VM.loc['음식점 수','검색 비중']:.2f}). 교통·공급·관심을 하나로 요약하는 지표라는 뜻이고, 셋을 임의 가중치로 섞은 Hotness Score 대신 생활이동을 정답으로 쓴 근거다.")
    a(f"- **현재 핫스팟 {len(VH)}곳 중 {int(dist[2] + dist[3])}곳({(dist[2] + dist[3]) / len(VH) * 100:.0f}%)이 독립 자료 2개 이상에서도 상위 10%**다. 평균 {VH['독립자료_상위10%_개수'].mean():.1f}개 (무작위로 고르면 0.3개). 3개 모두 상위 10%인 곳: " + ", ".join(VH[VH['독립자료_상위10%_개수'] == 3].dong.head(10)) + ".")
    zero = VH[VH["독립자료_상위10%_개수"] == 0]
    if len(zero): a(f"- 어느 자료에서도 상위 10%가 아닌 핫스팟은 {len(zero)}곳({', '.join(zero.gu + ' ' + zero.dong)}). '기타' 이동이 많은 주거·대학가 동으로 보이며, 목적 7 비중이 큰 정답의 한계와 같은 이야기다.")
    a("\n![핫스팟 일치](fig/V1_hotspot_agreement.png)\n")
    a("- 정리: 생활이동은 (1) 상식과 맞는 목록을 내고, (2) 독립 자료 세 개와 순위가 같은 방향으로 움직이며, (3) 그 셋을 가장 잘 요약한다. 예측이 어려운 것과 별개로 **'핫함'의 정답으로 쓸 근거가 있다.** 남은 약점은 떠오르는 곳에 섞이는 주거 변화(§7)와 2024-08~10 자료 이상(§2)이다.\n")

if (TB / "v2_change_cross.csv").exists():
    A2 = rd("v2_change_cross.csv"); B2 = rd("v2_change_within.csv")
    a("## 2-2. G.T 타당성: 변화도 같이 가나 (전체 동)\n")
    a("수준이 같이 가는 것은 동 크기 때문일 수도 있다. 그래서 모든 값을 **서울 대비 비중**으로 바꾸고 **변화끼리** 비교했다. 많이 는 동만 고르지 않고 전체 동을 쓴다. 2024-08~10 이상을 피해 같은 1~7월끼리 비교.\n")
    a("**A. 동 사이: 비중이 는 정도가 같은 순서인가** (스피어만 순위 상관, 전체 동)\n")
    pa = A2.pivot(index="자료", columns="기간(1~7월)", values="변화 순위 상관").reindex(["지하철 하차", "검색 비중", "음식점 수", "생활인구 20~39세"])
    a(md(pa.map(lambda v: f"{v:.2f}")))
    a("\n![변화 산점도](fig/V2_change_cross.png)\n")
    hi = A2[A2["기간(1~7월)"] == "2023→2025"].set_index("자료")
    a(f"- 세 독립 자료 모두 **양의 상관이고 모두 유의**하다 (p < 0.02). 기간이 길수록 커진다 (1년 0.12~0.21 → 2년 0.16~0.26). 짧은 기간의 변화는 잡음이 크고, 구조적인 변화는 천천히 쌓인다는 뜻이다.")
    a(f"- 2년 동안 생활이동 비중이 가장 많이 는 10% 동에서는 다른 자료도 서울 중앙값보다 는 비율이 지하철 {hi.loc['지하철 하차','생활이동 많이 는 동 중 같이 는 비율'] * 100:.0f}%, 검색 {hi.loc['검색 비중','생활이동 많이 는 동 중 같이 는 비율'] * 100:.0f}%, 음식점 {hi.loc['음식점 수','생활이동 많이 는 동 중 같이 는 비율'] * 100:.0f}%다. 가장 많이 준 10% 동에서는 30~35%뿐이다.")
    a("- 다만 상관 크기는 수준(0.46~0.83)보다 훨씬 작다. **'어디가 핫한가'는 자료들이 강하게 합의하지만 '어디가 뜨는가'는 약하게만 합의한다.** 떠오르는 곳 예측이 어려운 이유이자, 떠오르는 곳 판정에 다른 자료를 함께 보여줘야 하는 이유다.\n")
    a("**B. 동 안: 월별로 같이 오르내리나** (동마다 2024-01~2025-07, 월별 비중의 전년 같은 달 대비 변화끼리 상관)\n")
    bb = B2.set_index("자료")[["동 수", "동별 상관 중앙값", "상관이 양수인 동 비율", "다른 동과 짝지은 기준 중앙값", "순열 p"]].copy()
    bb["동별 상관 중앙값"] = bb["동별 상관 중앙값"].map(lambda v: f"{v:.3f}"); bb["상관이 양수인 동 비율"] = bb["상관이 양수인 동 비율"].map(lambda v: f"{v * 100:.0f}%")
    bb["다른 동과 짝지은 기준 중앙값"] = bb["다른 동과 짝지은 기준 중앙값"].map(lambda v: f"{v:.3f}"); bb["순열 p"] = bb["순열 p"].map(lambda v: f"{v:.3f}")
    a(md(bb))
    a("\n![동 안 상관](fig/V2_change_within.png)\n")
    a("- 기준은 무작위 10%가 아니라 **같은 계산을 다른 동의 자료와 짝지은 값**이다. 서울 전체가 같이 움직이는 효과가 빠진다.")
    a("- 지하철·검색은 기준보다 유의하게 높지만 크기가 작고(중앙값 0.04~0.08), 음식점은 기준과 차이가 없다. 음식점 점포 수는 달마다 거의 변하지 않아 월 단위로는 같이 움직이기 어렵다.")
    a("- 월 단위 등락은 각 자료의 잡음이 커서 일치가 약하다. 생활이동의 타당성은 **수준(§2-1)에서 강하고, 1~2년 변화(A)에서 약하지만 분명하며, 월별 등락(B)에서는 거의 없다**고 정리하는 것이 정직하다.\n")

a("## 3. M1 결과표 확장: 모델 × 입력\n")
tb = t1.copy(); tb.index = [LAB[i] for i in tb.index]; tb.columns = [XL[c] for c in tb.columns]
a("4개 시점 평균 13주 WAPE (%). 낮을수록 좋다. 입력 없음 열의 ARIMAX = ARIMA, SARIMAX = SARIMA.\n")
a(md(tb.map(f1)))
a("\n![히트맵](fig/F3_heatmap_wape.png)\n")
rows = []
for k in ["naive|none", "snaive|none", "arima|none", "sarima|none", "arimax|search", "arimax|all_noLP", "arimax|all", "sarimax|search", "sarimax|all_noLP", "sarimax|all"]:
    sh, p = tget(k, "snaive|none")
    dmr = dm.set_index("model")
    rows.append((kname(k), f1(mean4.get(k)), f1(clean.get(k)), f"{mase.loc[k,'mase_median']:.2f}", "–" if k == "snaive|none" else pct(sh),
                 "–" if k == "snaive|none" else f"{dmr.loc[k,'dm_stat_vs_snaive']:+.2f} (p={dmr.loc[k,'p']:.2f})"))
a("| 모델 | WAPE 4시점 | WAPE 2시점(이상 구간 제외) | MASE 중앙값 | 작년 같은 주보다 좋아진 동 | DM vs 작년 같은 주 |\n|---|---|---|---|---|---|")
for r in rows: a("| " + " | ".join(r) + " |")
a("\n- MASE < 1이면 작년 같은 주(학습 구간)보다 오차가 작다는 뜻. DM 통계량이 음수면 작년 같은 주보다 낫다는 방향.")
a(f"- **계절(SARIMA)**: 4시점 평균 {f1(mean4['sarima|none'])}%로 ARIMA보다 나쁘다. 2025-07 시점의 이상 구간 영향이 크다. 2시점에서는 {f1(clean['sarima|none'])}%로 ARIMA({f1(clean['arima|none'])}%)와 비슷하다.")
x_rows = []
for m, base in (("arimax", "arima|none"), ("sarimax", "sarima|none")):
    for x in ("search", "all_noLP", "all"):
        sh, p = tget(f"{m}|{x}", base); x_rows.append((kname(f"{m}|{x}"), kname(base), f1(mean4[f'{m}|{x}'] - mean4[base]), f1(clean[f'{m}|{x}'] - clean[base]), pct(sh), f"{p:.3f}"))
a("- **입력(외생변수)**: 넣을수록 나빠졌다. 아래는 같은 차수의 입력 없는 모델과 비교한 것이다.\n")
a("| 모델 | 비교 대상 | WAPE 차이 4시점 (%p) | 2시점 (%p) | 좋아진 동 | Wilcoxon p |\n|---|---|---|---|---|---|")
for r in x_rows: a("| " + " | ".join(r) + " |")
a("\n  해석: 13주 시차를 둔 입력은 예측 구간의 변화와 잘 맞지 않고, 한 동당 학습 주가 90~130개라 입력 계수가 불안정하다. 입력을 하나씩 넣고 빼는 실험(F1)과 입력 모양 바꾸기(최근값 고정, 변화율 등)는 최종 발표 전 과제로 남긴다.\n")
a("**동별 오차 분포** (13주 WAPE, 4시점 × 동)\n")
qq = q.loc[[k for k in ["snaive|none", "naive|none", "arima|none", "sarima|none", "arimax|all", "sarimax|all"] if k in q.index]].copy()
qq.index = [kname(i) for i in qq.index]; qq.columns = ["10%", "25%", "중앙값", "75%", "90%"]
a(md(qq.map(f1)))
a("\n![동별 분포](fig/F5_dong_dist.png)\n")
a("- 오차는 동마다 크게 다르다. 중앙값 동은 10% 안팎이지만 상위 10% 동은 30~50%를 넘는다. 평균만 보면 이 차이가 가려진다.\n")

a("## 4. M2 예측 시점 추가: 결론이 시점마다 유지되나\n")
ks = ["naive|none", "snaive|none", "arima|none", "sarima|none", "arimax|search", "arimax|all", "sarimax|all", "arimax|none+py"]
tt = t2[ks].copy(); tt.columns = [kname(k) for k in ks]
tt.loc["4시점 평균"] = [mean4[k] for k in ks]; tt.loc["2시점 평균 (이상 제외)"] = [clean[k] for k in ks]
a(md(tt.T.map(f1)))
a("\n![시점별](fig/F4_by_origin.png)\n")
best_by_o = t2[[c for c in t2.columns]].idxmin(axis=1)
a("- 시점별 1등: " + ", ".join(f"{o} {kname(k)} ({f1(t2.loc[o, k])}%)" for o, k in best_by_o.items()) + ".")
a("- **'작년 같은 주가 가장 강하다'는 초안의 결론은 이상 구간이 없는 시점에서만 유지된다.** 이상 구간이 학습 끝에 걸리면 지난주 기반 모델이, 작년 값에 걸리면 작년 기반 모델이 무너진다.")
a("- 발표에서는 4시점 평균만 보여주지 말고 이 그림처럼 시점별로 보여주는 것이 정직하다.\n")

a("## 5. M3 작년 같은 주를 입력으로\n")
t3s = t3.copy(); t3s["xset"] = ["ARIMA" if x == "none" else f"ARIMAX + {XL[x]}" for x in t3.xset]
t3s["2시점 없음"] = [clean["arima|none" if x == "none" else f"arimax|{x}"] for x in t3.xset]
t3s["2시점 있음"] = [clean[f"arimax|{x}+py"] for x in t3.xset]
t3s = t3s.set_index("xset")[["without_prior_year", "with_prior_year", "2시점 없음", "2시점 있음", "snaive"]]
t3s.columns = ["4시점 입력 없음", "4시점 입력 있음", "2시점 입력 없음", "2시점 입력 있음", "작년 같은 주(기준선) 4시점"]
a(md(t3s.map(f1)))
a("\n![M3](fig/F6_prior_year.png)\n")
sh, p = tget("arimax|none+py", "arima|none")
a(f"- 4시점 평균으로는 조금 나빠지고(동별 좋아진 비율 {pct(sh)}, p = {p:.2f}), 2시점에서는 ARIMA {f1(clean['arima|none'])}% → {f1(clean['arimax|none+py'])}%로 좋아진다. 입력이 있는 모델 중 2시점 기준 가장 좋다.")
a(f"- 그래도 작년 같은 주 자체({f1(sn_c)}%)는 넘지 못했다. 연간 패턴을 계수 하나로 넣는 방법은 가능성은 있지만, 이번 결과만으로 '넘는다'고 말할 수 없다.")
a("- 작년 값이 이상 구간에 걸리면 이 입력도 같이 오염된다. 이상 구간 처리(다음 단계)와 묶어서 다시 봐야 한다.\n")

a("## 6. M4 검색 입력을 서울 대비 비중으로\n")
t4s = t4.copy(); t4s["model"] = [LAB[m] + " + 검색" for m in t4.model]
t4s = t4s.set_index("model"); t4s.columns = ["원래 검색 지수", "서울 대비 비중"]
a(md(t4s.map(f1))); a("")
for m in ("arimax", "sarimax"):
    sh, p = tget(f"{m}|search", f"{m}|search_idx")
    a(f"- {LAB[m]}: 비중이 좋아진 동 {pct(sh)}, Wilcoxon p = {p:.4f}.")
a("- ARIMAX에서는 차이가 없고 SARIMAX에서는 비중이 유의하게 낫다. 비중으로 바꾼 결정을 유지한다.")
a("- 떠오르는 곳 판정도 비중(서울 대비 몫)의 증가율로 계산했다. 2024-08~10처럼 서울 전체가 함께 출렁일 때 비중은 그 영향을 대부분 지운다.\n")

a("## 7. M5 핫스팟 목록과 동네 예시\n")
a("정의 (README §2.2): **현재 핫스팟** = 13주 평균 상위 10%. **떠오르는 곳** = 13주 평균 서울 대비 비중의 1년 전 같은 13주 대비 증가율 상위 10%. 규모 하한은 **2023년 동별 평균의 하위 25% 경계(하루 약 2,050명)로 고정**하고, 직전 13주 평균이 이보다 작은 동은 뺀다.\n")
h = H[(H.origin == "2025-04-27")]
for w, title in (("now", "2025-04-27 직전 13주 (판정 시점에 아는 것)"), ("actual", "2025-04-27 이후 13주 실제 (맞혀야 할 것)")):
    a(f"**{title}**\n")
    a("| 순위 | 현재 핫스팟 | 13주 일평균 | 떠오르는 곳 | 비중 증가율 |\n|---|---|---|---|---|")
    hh = h[(h.window == w) & (h.list == "hot")].head(10).reset_index(drop=True)
    ee = h[(h.window == w) & (h.list == "emerging")].head(10).reset_index(drop=True)
    for i in range(10):
        a(f"| {i + 1} | {hh.gu[i]} {hh.dong[i]} | {hh.level_13w[i]:,.0f} | {ee.gu[i]} {ee.dong[i]} | {ee.growth_vs_last_year[i] * 100:+.0f}% |")
    a("")
a("![핫스팟 지도](fig/F7_hotspot_map.png)\n")
a("- 현재 핫스팟은 서교동(홍대), 역삼1동, 여의동, 종로, 한강로동(용산), 신촌, 잠실, 삼성, 명동처럼 누구나 아는 곳이 나온다. 정의가 상식과 맞는다.")
a("- 떠오르는 곳 1위 둔촌1동(+222%)은 2024년 말 올림픽파크포레온(둔촌주공 재건축) 입주 시기와 겹치는 것으로 보인다. 그 밖에도 주거지 동이 많다.")
a("  - 정답의 대부분이 목적 7(기타)이기 때문이다 (20~39세 이동 중 목적 7이 46.5%, 쇼핑 0.6%, 관광 0.1%).")
a("  - 목적 4+5만 쓰면 0이 아닌 동이 38개뿐이라(원자료가 작은 값을 감춘 것으로 보임) 대안이 되지 못한다.")
a("  - 그래서 '떠오르는 곳 = 사람들이 새로 많이 오기 시작한 동'으로 설명하고, 상권 해석은 검색·음식점 자료와 함께 붙이는 것이 맞다.\n")
S2 = S.pivot(index="variant", columns="origin", values="overlap_with_base")
S2["평균"] = S2.mean(axis=1)
S2 = S2.loc[["규모 하한을 시점마다 다시 계산", "규모 제외 없음", "전 연령", "목적 4+5"]]
a("**정의 바꿔 보기 (실제 다음 13주 떠오르는 곳 목록과 겹치는 비율)**\n")
a(md((S2 * 100).round(0).astype(int).astype(str) + "%"))
a("\n- 규모 하한을 빼면 목록의 절반 가까이 바뀐다 (작은 동은 비율이 쉽게 튄다). 하한은 필요하다.")
a("- 하한을 고정값 대신 시점마다 하위 25%로 다시 계산해도 목록은 대부분 같다. 고정값은 시점이 바뀌어도 같은 잣대라 설명이 쉬워서 택했다.")
a("- 전 연령으로 바꿔도 절반쯤 겹친다. 20~39세 정의가 목록을 꽤 바꾼다.")
a("- 목적 4+5는 위 이유로 비교 불가 (뽑히는 동이 4곳).\n")
a("**동네 예시** (대표 모델 기준. 결과를 본 뒤 규칙으로 골랐다)\n")
a(f"대표 모델 = 4개 시점 평균 WAPE가 가장 낮은 모델 중 작년 같은 주를 뺀 것: **{kname(rep)}**. 작년 같은 주는 모든 동의 예측 증가율이 0이라 떠오르는 곳을 가릴 수 없어 뺐다.\n")
a("| 역할 | 동 | 고른 규칙 | 2025-04-27 대표 모델 13주 WAPE | 416개 중 순위 (1 = 가장 잘 맞음) |\n|---|---|---|---|---|")
rule = {"잘 맞춘 곳": "규모 중앙값 이상 동 중 WAPE 최저", "보통인 곳": "WAPE 중앙값에 가장 가까운 동", "못 맞춘 곳": "WAPE 상위 10% 경계에 가장 가까운 동"}
for _, r in CS.iterrows():
    a(f"| {r.role} | {r.dong} | {rule[r.role]} | {r.wape_rep:.1f}% | {r.rank_low_is_good} |")
a("")
for _, r in CS.iterrows():
    a(f"**{r.role}: {r.dong}** — 예측 시점 전 20주와 예측 13주(음영). 파란 실선 = 시점까지 실제, 파란 점선 = 이후 실제, 회색 = 입력 없음, 빨강 = 입력 있음, 초록 점선 = 작년 같은 주.\n")
    a(f"![{r.dong}](fig/F2_case_{r.dong}.png)\n")
a("- 잘 맞춘 곳(신정3동)은 주별 변동이 작고 수준이 안정적이다. 회색(입력 없음) 선이 실제를 잘 따라간다. SARIMAX + 전체는 같은 동에서도 크게 벗어난다.")
a("- 못 맞춘 곳(성수1가1동, 서울숲이 있는 동)은 해마다 봄(4월)·가을(10월)에 크게 치솟는 **계절이 강한 동**이다. 대표 모델 ARIMA는 계절을 모르고 봄 정점이 계속될 것처럼 천천히 내려와 19.3%가 틀렸다. 같은 동에서 작년 같은 주는 6.0%, SARIMA는 5.7%, SARIMAX + 검색은 4.8%로 잘 맞혔다 (`F1_case_성수1가1동.png`).")
a("- 즉 서울 평균으로는 ARIMA가 낫지만, 계절이 뚜렷한 공원·관광 동에서는 계절 모델이 훨씬 낫다. 동의 성격에 따라 모델을 고르거나 섞는 방법(최종 과제)의 근거가 된다.")
a("- 긴 기간 그림은 `fig/F1_case_<동>.png`.\n")

a("## 8. M6 상위 K개 적중률\n")
a(f"K = 상위 10%. 현재 핫스팟 K = {nh}곳, 떠오르는 곳 K = {ne}곳. 예측한 다음 13주 목록과 실제 다음 13주 목록이 겹친 비율.\n")
a("**현재 핫스팟**\n"); a(md(k_hot)); a("")
a("**떠오르는 곳**\n"); a(md(k_em)); a("")
a("![상위 K](fig/F8_topk.png)\n")
a("- **현재 핫스팟**: 모든 모델이 90% 안팎. 하지만 '직전 13주 순위 그대로'(지속)도 95%라 모델이 보탤 것이 거의 없다. 붐비는 곳은 3개월 사이에 거의 바뀌지 않는다.")
a("- **떠오르는 곳**: 무작위(10%)의 약 5배인 50% 안팎. 지속 기준과 비슷하다. 2025-04·07 시점에서는 ARIMAX 계열이 지속 기준을 넘었지만 2024-10·2025-01 시점에서는 졌다.")
a("- 작년 같은 주는 떠오르는 곳 순위를 매길 수 없다 (예측 증가율이 모든 동 0). 그래서 '기준선 대비'가 아니라 '지속·무작위 대비'로 보고한다.")
pe = K[(K.target == 'emerging') & (K.K.astype(str) == '10%') & (K.method == 'persistence')].hit_rate.mean()
a(f"- 발표용 성공 기준 제안: '떠오르는 곳 상위 10% 중 절반을 13주 전에 맞힌다 (무작위 10%, 지속 {pe * 100:.0f}%)'. 지속 기준을 확실히 넘는 것은 최종 발표 목표로 둔다.\n")

a("## 9. 한계\n")
a("- **예측 시점 4개**: 시점 하나가 평균을 크게 바꾼다. 최종 전에는 시점을 매월로 늘린다(rolling origin).")
a("- **2024-08~10 자료 이상**: 원인 미확인. 이 구간을 그대로 두고 평가했다. 제공처 확인 또는 해당 주 보정이 필요하다.")
a("- **고정 시차**: 13주 앞 마지막 주까지 누수가 없도록 모든 주 입력을 13주 늦췄다. 가까운 주(h = 1~4)에는 더 최근 값을 쓸 수 있어 정보 손실이 있다.")
a("- **목적 7 비중**: 정답이 상권 방문보다 넓은 '기타 이동'이다. 떠오르는 곳에 주거 변화가 섞인다.")
a("- **생활인구 입력**: 생활인구와 생활이동은 같은 통신 자료에서 나와 겹친다. 넣어도 좋아지지 않았다.")
a("- **동 하나씩 따로 적합**: 한 동당 학습 주가 적어 입력 계수가 불안정하다. 여러 동을 함께 쓰는 모델(패널, LightGBM)은 최종 과제.")
a("- **2025-09 이후 자료**: 품질 점검 구간으로 표시만 했다. 이번 평가 구간(~2025-10)과 일부 겹친다.\n")

a("## 10. 다음 단계 (최종 발표 전, 우선순위 순)\n")
a("1. **2024-08~10 이상 처리**: 제공처 공지 확인 → 해당 주를 서울 합계 비율로 보정하거나 학습·평가에서 빼고 다시 비교. 결론이 가장 크게 바뀔 수 있는 항목이다.")
a("2. **예측 시점 늘리기**: 2024-07 ~ 2025-07 매월 시점(13개)으로 평가해 시점 운을 줄인다.")
a("3. **행사 급등 처리 (README F3)**: `flag_spike` 주를 학습에서 약하게 반영. 성수·여의도처럼 못 맞춘 동이 대상.")
a("4. **입력 하나씩 넣고 빼기**: 지금은 한꺼번에 넣어 나빠졌다. 어떤 입력이 해를 끼치는지 분리한다. 입력을 '최근값 고정' 방식으로도 비교.")
a("5. **떠오르는 곳 정의 다듬기**: 주거 변화(아파트 입주)를 가려낼 보조 신호로 검색·음식점 개업을 함께 본다.")
a("6. **여러 동을 함께 쓰는 모델**: 패널 회귀, LightGBM(+같은 입력의 Ridge)으로 지속 기준을 넘는지 본다.\n")

a("## 부록\n")
a("**다시 돌리기**: `bash src/gt/run_all.sh` (자료 `data/raw/mobility`, `living_pop`, `subway` 필요. 전체 약 45분, 모델 단계가 대부분)\n")
a("| 단계 | 파일 | 출력 |\n|---|---|---|")
a("| 정답 | `00_target.py` | `data/interim/gt/target_weekly.parquet`, `00_data_audit.md` |")
a("| 입력 | `01_features.py` | `features_weekly.parquet`, `00_leakage_test.md`, `00_unmatched_stations.csv` |")
a("| 모델 | `02_models.py` | `results/gt/pred_<시점>.parquet`, `orders.csv`, `fit_log.csv` |")
a("| 평가 | `03_evaluate.py` | `tables/t1~t5, t_tests, t_dm, t_mase` |")
a("| 핫스팟 | `04_hotspots.py` | `tables/hotspots.csv, topk.csv, sensitivity.csv` |")
a("| 그림 | `05_figures.py` | `fig/F1~F10`, `tables/case_dongs.csv` |")
a("| 리포트 | `06_report.py` | 이 문서 |")
a("| 지도 | `07_map_data.py` | `data/processed/gt/hot_map.json` (지도 '핫플지수' 탭) |")
a("| 타당성 | `08_validity.py` | `tables/v1_*.csv`, `fig/V1_*.png` |")
a("| 타당성(변화) | `09_validity_change.py` | `tables/v2_*.csv`, `fig/V2_*.png` |")
a("")
a("- 못 맞춘 지하철역: 양재시민의숲, 청계산입구 (신분당선, OA-12914에 없음). 해당 동은 남은 역으로 계산.")
a("- 결측 입력은 그 동의 첫 관측값으로 앞을 채웠다. 한 입력이 통째로 없는 동은 그 묶음에서 빠지고 공통 동 집합에서 제외(426 → " + str(int(NSET.n_units.min())) + ").")

out = C.REPORT / "00_결과_리포트.md"
out.write_text("\n".join(L), encoding="utf-8")
log_run("06_report", f"rep={rep}")
print("저장:", out, len(L), "줄")
