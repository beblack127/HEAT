# 지표 명세서 (feature_spec)

> 코드와 보고서가 같은 정의를 쓰도록 하는 단일 기준 문서. 수식을 바꾸면 이 파일을 먼저 고치고 `experiments/log.csv`의 `spec_version`을 올린다.
> spec_version: **2.2** (2026-11-05)
> 변경 이력: v2.0 예측 타깃을 비거주 방문 성장으로 변경, FVI는 기술 지표로, 이동(M)·검색(S) 피처 그룹 추가 · v2.1 선후 관계 양방향 검증(§14), 프랜차이즈 진입률 추가 · v2.2 정상성 확인, 사전백색화 교차상관, 국소 투영 식(§14.5)
>
> 구성: §1–§5 F&B 지표 · §6 FVI · §7 이동(M) · §8 검색(S) · §9 타깃 · §10 파생 피처 · §11 실험용 피처 그룹 · §12 사후 유형 · §13 출력 스키마 · §14 선후 관계 분석 정의

## 0. 공통 규칙

| 항목 | 규칙 |
|---|---|
| 키 | `dong_code` (기준 행정동 코드 = **이동 데이터의 코드 체계**, Phase 1에서 고정), `ym` (YYYY-MM, 월말 기준) |
| 점포 집합 | `stores_clean`에서 `include == 1`인 점포. `bulk_closure` 처리 방식은 패널 버전(`v_keep`, `v_drop`)으로 구분 |
| 영업 점포 수 | S_t = #{ 인허가일 ≤ t월말 AND (폐업일 없음 OR 폐업일 > t월말) } |
| 개업·폐업 | O_t = #{ 인허가일 ∈ t월 }, C_t = #{ 폐업일 ∈ t월 } |
| 시점 규칙 | t월 값은 t월 말까지 실제로 알 수 있던 정보만 사용 |
| 분모 0 | S_{t-1} = 0이면 비율은 결측(NaN). 0으로 채우지 않는다 |
| 작은 수 안정화 | 비율 지표는 `_ma3`(3개월 이동합으로 계산) 과 `_eb`(Empirical Bayes 축소) 두 버전을 만든다. 모델 기본값은 `_eb` |
| 이상치 | 비율 지표는 월별 횡단면 1%·99% winsorize (`_w` 접미사 없이 최종 컬럼에 적용) |
| 범주 접미사 | `_cafe`, `_trend`, `_fr`(프랜차이즈), `_indie`(독립) 등 같은 정의를 부분집합에 적용한 버전 |

### Empirical Bayes 축소 (비율 지표 공통)

동 i의 사건 수 x_i, 분모 n_i, 같은 구(區) 동들의 평균 비율 m, 동 간 분산 추정치 s²에 대해:

```
raw_i  = x_i / n_i
w_i    = s² / (s² + m / n_i)
eb_i   = w_i * raw_i + (1 - w_i) * m
```

- m과 s²는 같은 달, 같은 구의 동들로 추정한다(method of moments). s² < 0이면 0으로 둔다.
- 분모가 작은 동일수록 구 평균 쪽으로 더 많이 당겨진다(Clayton & Kaldor 1987의 소지역 비율 평활 원리).

---

## 1. 재고·흐름 (G1)

| ID | 컬럼 | 수식 | 창 | 비고 |
|---|---|---|---|---|
| F01 | `stock` | S_t | 월말 | 모든 비율의 분모. `stock_cafe`, `stock_trend`, `stock_fr` 등 범주 버전 포함 |
| F02 | `entry_rate` | O_t / S_{t-1} | 1개월 | `_ma3`: ΣO(t-2..t) / S_{t-3} |
| F03 | `exit_rate` | C_t / S_{t-1} | 1개월 | 동일 |
| F04 | `net_rate` | (O_t − C_t) / S_{t-1} | 1개월 | `net_rate_12m` = Σ(O−C)(t-11..t) / S_{t-12} → FVI 성분 c1 |
| F05 | `churn_rate` | (O_t + C_t) / S_{t-1} | 1개월 | `_ma3` 동일 |
| F05b | `replace_share` | t월 폐업 중 `replaced == 1` 비율 | 3개월 | 보조. 양수·양도성 회전 비중 |
| F05c | `growth_churn_type` | F04_12m, F05_12m을 월별 서울 중앙값 대비 높음/낮음으로 2×2 → {뜨는중, 안정성장, 교체과포화, 정체} | 12개월 | 범주형 |

## 2. 업력·생존 (G2)

| ID | 컬럼 | 수식 | 창 | 비고 |
|---|---|---|---|---|
| F06 | `age_active_mean` | 영업 점포들의 (t월말 − 인허가일) 평균, 개월 | 월말 | 중앙값 버전 `age_active_median` |
| F07 | `tenure_closed_mean` | 최근 12개월 폐업 점포들의 (폐업일 − 인허가일) 평균, 개월 | 12개월 | 폐업 5건 미만이면 NaN |
| F08 | `cci_type` | F06, F07을 월별 서울 전체 평균과 비교한 2×2 유형 | 12개월 | 공식 상권변화지표의 유형 대응 규칙을 데이터 정의서로 확인 후 동일하게 적용 |
| F09 | `surv6` | (t−12 ~ t−7월 개업 점포 중 개업+6개월 시점에 영업 중) / (같은 기간 개업 점포) | 6개월 코호트 | 개업 5건 미만이면 NaN → EB로 보완. FVI 성분 c2 |

시점 확인: F09의 코호트는 가장 늦은 개업이 t−7월이므로 개업+6개월이 t−1월 이내에 끝난다. t월 말 기준으로 관측이 완료되어 있다.

## 3. 업종 구성 (G3)

| ID | 컬럼 | 수식 | 창 | 비고 |
|---|---|---|---|---|
| F10 | `trend_share` | stock_trend / stock | 월말 | `is_trend == 1` 범주(카페, 주점, 양식, 외국음식 등). 변화량 `trend_share_d12` = t − (t−12) |
| F11 | `indie_cafe_entry_share` | 최근 3개월 카페 개업 중 `is_franchise == 0` 비율 | 3개월 | 카페 개업 3건 미만이면 NaN → EB. 함께 만들 것: 독립 카페 진입률 `indie_cafe_entry_rate` = 최근 3개월 독립 카페 개업 / stock_cafe_{t−3}, 프랜차이즈 진입률 `franchise_entry_rate` = 최근 3개월 프랜차이즈(카페·외식) 개업 / stock_{t−3}. 둘은 선후 관계 업종별 분리 검증(§14)에 쓴다 |
| F12 | `franchise_share` | stock_fr / stock (카페·외식 대상) | 월말 | 상권분석서비스 점포-행정동의 프랜차이즈 점포수로 검증 |
| F13 | `diversity_entropy` | −Σ p_k ln p_k / ln K, p_k = 범주 k 점포 비율 | 월말 | K = 매핑표의 포함 범주 수(고정) |

## 4. 공간·규모 (G4)

| ID | 컬럼 | 수식 | 창 | 비고 |
|---|---|---|---|---|
| F14 | `nbr_net_rate` | 이웃 동 집합 N(i)의 `net_rate_eb` 평균 (queen contiguity, 행 표준화) | 1개월 | `nbr_net_rate_12m`, `nbr_indie_cafe_entry_rate` 도 함께. 이웃 없는 동은 k=3 최근접 |
| F15 | `net_area` | t월 개업 점포 면적 합 − 폐업 점포 면적 합 (㎡) | 1개월 | `net_area_ratio` = net_area / 영업 점포 총면적. 면적 결측은 동·범주별 중앙값 대체, 상위 1% 절단 |

## 5. 매출 (G5) — 분기 데이터, 월로 as-of 결합

외식 업종 집합(매핑표 `source == 추정매출`, `include == 1`)만 합산한다.

| ID | 컬럼 | 수식 | 비고 |
|---|---|---|---|
| — | `sales_q` | 분기 q 외식 매출 합 | 중간 산출물 |
| — | `stock_q_mean` | 분기 q 세 달의 S_t 평균 | 인허가 기반 |
| F16 | `sales_per_store_yoy` | (sales_q / stock_q_mean) / (sales_{q−4} / stock_{q−4}_mean) − 1 | FVI 성분 c3. 전년 동분기 대비로 계절성 제거 |
| F17 | `supply_demand_type` | 점포수 전년 동분기 대비 증감 부호 × F16 부호 → {수요주도성장, 과잉공급, 구조조정회복, 쇠퇴, 변화없음} | 변화 폭이 ±2% 이내인 축은 "변화없음"으로 처리 |
| F18 | `weekend_share` | 주말 매출 / 전체 외식 매출 | 변화량 `_d4`(전년 동분기 대비) 함께 |
| F19 | `young_share` | (20대 + 30대 매출) / 연령대 매출 합 | 변화량 `_d4` 함께 |

**월 결합 규칙**: 분기 q의 값은 `release_month(q) = 분기 마지막 달 + SALES_RELEASE_LAG_MONTHS`(기본 3) 이후의 모든 월에 붙는다. 각 월에는 그 시점에 공표된 가장 최근 분기 값이 들어간다. `sales_age_months`(값이 몇 개월 묵었는지)도 피처로 넣는다.

---

## 6. F&B 활력지수 (FVI) — 기술 지표, 예측 타깃 아님

F&B 상태를 한 값으로 요약한다. 용도는 (1) "F&B 개별 지표 vs 요약 지수" 비교 실험(E1-fvi), (2) 사후 유형 분류의 F&B 축(§12), (3) 팀 Hotness Score의 F&B 성분 후보.

| 컬럼 | 정의 |
|---|---|
| `fvi` | (z(net_rate_12m_eb) + z(surv6_eb) + z(sales_per_store_yoy)) / 3. z는 같은 달 동 횡단면 표준화 |
| `fvi_lite` | (z(net_rate_12m_eb) + z(surv6_eb)) / 2. 매출이 없는 구간용 |
| `fvi_d3`, `fvi_d6` | fvi_t − fvi_{t−3}, fvi_t − fvi_{t−6} |

민감도: PCA 제1주성분 가중 FVI(각 origin의 학습 구간에서만 적합).

---

## 7. 이동 데이터 (M 그룹)

### 7.1 입력 스키마 — `data/external/mobility_monthly.csv`

| 컬럼 | 설명 |
|---|---|
| `dong_code`, `ym` | 키 |
| `arr_total` | 월 도착 합계 |
| `arr_home`, `arr_commute` | 귀가, 출근 목적 도착 |
| `arr_<목적>` | 나머지 목적별 도착 (기타, 쇼핑, 관광 등. 실제 목적 이름은 Phase 0 감사 기준) |
| `arr_visit_weekend` | (선택) 주말·공휴일의 비거주·비통근 도착 |
| `shift_flag` | 데이터 수준 변동 월이면 1 (제안서 EDA의 회색 구간) |

### 7.2 정의

```
V_t              = arr_total − arr_home − arr_commute          # 비거주·비통근 방문
visit_share_t    = V_t / Σ_dong V_t                            # 서울 전체 대비 비중 → 도시 공통 변동 제거
visit_sh3_t      = mean(visit_share_{t−2..t})                  # 3개월 평균
home_share_t     = arr_home_t / Σ_dong arr_home_t
home_sh3_t       = mean(home_share_{t−2..t})
```

### 7.3 M 그룹 피처

| 컬럼 | 수식 | 비고 |
|---|---|---|
| `visit_g3`, `visit_g6` | log(visit_sh3_t) − log(visit_sh3_{t−3}), …_{t−6} | 과거 방문 성장(모멘텀) |
| `visit_level_pct` | visit_sh3_t의 그 달 백분위 | 이미 큰 동인지 |
| `visit_g3_lag3` | visit_g3의 3개월 시차 | |
| `home_g3`, `home_g6` | 귀가 비중의 같은 방식 성장률 | **거주 통제 (H3)** |
| `weekend_visit_share` | arr_visit_weekend / V | 방문객형 상권 신호 (선택) |
| `livpop_*`, `subway_*` | 생활인구·지하철 월 집계의 성장률 | 있으면 추가 (선택) |

**변동 구간 규칙**: 계산 창(t−6..t)에 `shift_flag == 1`인 월이 있으면 해당 피처는 NaN. 보간하지 않는다.

---

## 8. 검색량 (S 그룹)

### 8.1 입력 스키마 — `data/external/search_monthly.csv`

| 컬럼 | 설명 |
|---|---|
| `dong_code`, `ym` | 키 |
| `search_index` | 동 대표 키워드(들)의 월 검색 지수 |
| `keyword_set` | 사용한 키워드 (문서화용) |

데이터랩 지수는 요청 단위의 상대값이므로, 팀원이 동 간 비교가 가능하도록 정규화한 값을 받는다. 키워드 매칭이 안 된 동은 NaN.

### 8.2 S 그룹 피처

| 컬럼 | 수식 |
|---|---|
| `search_g3`, `search_g6` | log(search_sh3_t) − log(search_sh3_{t−3}), …_{t−6}. search_sh3 = 검색 지수 3개월 평균 |
| `search_level_pct` | search_sh3_t의 그 달 백분위 |
| `search_g3_lag3` | search_g3의 3개월 시차 |

---

## 9. 타깃

| 컬럼 | 정의 |
|---|---|
| `y_h{h}` | log(visit_sh3_{t+h}) − log(visit_sh3_t), h = 1, 3, 6. **주 회귀 타깃은 `y_h3`** |
| `established_t` | visit_sh3_t가 그 달 상위 15% 이상이면 1 (이미 확립된 방문 집중지) |
| `resid_flag_h3` | home_g3_{t+3}가 그 달 상위 5% 이상이면 1 (거주형 증가) |
| `emerging_h3` | 아래 네 조건을 모두 만족하면 1 |
| `emerging_h3_nores` | 위에서 거주 필터(조건 4)만 뺀 버전 (H3 검증용) |

`emerging_h3` 조건 (origin t):
1. `established_t == 0`
2. `y_h3`가 origin t의 비확립 동들 중 상위 10%
3. 지속성: log(visit_sh3_{t+4}) − log(visit_sh3_t) > 0 AND log(visit_sh3_{t+5}) − log(visit_sh3_t) > 0
4. `resid_flag_h3 == 0`

**유효 관측**: t−2 ~ t+5 사이에 `shift_flag == 1`인 월이 없을 때만 타깃을 계산한다.
**민감도**: 확립 기준 상위 10·20%, 성장 기준 상위 5·20%, 지속성 한 달만, 거주 필터 상위 10%.
**팀 Hotness Score**: 팀이 별도 점수를 확정하면 `y_hot_h3`(3개월 뒤 점수 변화)를 대체 타깃으로 추가하고 같은 실험을 반복한다.

---

## 10. 모델 입력용 파생 피처

- F 지표: lag 1, 3, 6, 12 / 이동평균 3·6개월 / 변화량 d3, d12
- M·S 지표: lag 1, 3, 6 (이동 데이터가 짧아 12개월 시차는 쓰지 않음)
- 달력: `month`
- 정적: `gu_code`(범주형), 동 면적, 2023년 초 `stock`, 2023년 상반기 `visit_level_pct`
- 도시 공통 충격: 타깃이 서울 비중이라 대부분 제거됨. 선형 모형은 월 고정효과로 추가 흡수

## 11. 실험용 피처 그룹

```
M        = visit_g3, visit_g6, visit_level_pct, visit_g3_lag3, home_g3, home_g6,
           weekend_visit_share, (livpop_*, subway_*)  + 각 시차
M_nores  = M에서 home_* 제외                                  # E0-nores
F        = G1 ∪ G2 ∪ G3 ∪ G4 ∪ G5                              # F&B 지표 개별
  G1_flow  = F01–F05, F05b, F05c
  G2_age   = F06–F09
  G3_mix   = F10–F13
  G4_space = F14, F15
  G5_sales = F16–F19, sales_age_months
F_fvi    = fvi, fvi_d3, fvi_d6                                 # E1-fvi
S        = search_g3, search_g6, search_level_pct, search_g3_lag3
CAL      = month, gu_code, 정적 피처 (모든 실험에 공통 포함)
```

| 실험 | 피처 | 타깃 |
|---|---|---|
| E0 | M + CAL | y_h3, emerging_h3 |
| E1 | M + F + CAL | 〃 |
| E1-fvi | M + F_fvi + CAL | 〃 |
| E2 | M + S + CAL | 〃 |
| E3 | M + F + S + CAL | 〃 |
| E-F | F + CAL | 〃 |
| E0-nores | M_nores + CAL | y_h3, emerging_h3_nores |
| F1' | M + F04(net_rate) + CAL | 〃 |
| F1 ~ F5 | M + G1, + G2, + G3, + G4, + G5 (누적) | 〃 |

---

## 12. 사후 유형 분류 (Phase 8)

origin t 기준, 각 축의 신호는 그 달 횡단면 상위 1/3이면 1.

| 축 | 컬럼 | 신호 |
|---|---|---|
| 방문 | `sig_visit` | visit_g6 상위 1/3 |
| F&B | `sig_fnb` | fvi_d6 상위 1/3 |
| 관심 | `sig_attn` | search_g6 상위 1/3 (검색량 없으면 NaN) |

| `type` | 조건 |
|---|---|
| 동반 성장형 | sig_fnb = 1 AND sig_attn = 1 |
| F&B 주도형 | sig_fnb = 1 AND sig_attn = 0 |
| 관심 주도형 | sig_fnb = 0 AND sig_attn = 1 |
| 방문 단독형 | sig_fnb = 0 AND sig_attn = 0 |

검색량이 없으면 `F&B 동반형`(sig_fnb = 1) / `방문 단독형`(sig_fnb = 0) 2유형.
**지속률**: 유형별로 t+3 이후 visit_sh3가 origin 대비 높게 유지된 개월 수(최대 데이터 끝까지)의 평균. H5 검증에 쓴다.

## 13. 출력 스키마 요약

`data/processed/fnb_features.parquet`: `dong_code, ym, panel_version, F01…F19(접미사 버전 포함), fvi, fvi_lite, fvi_d3, fvi_d6, 파생 피처`
`data/processed/mobility_search_features.parquet`: `dong_code, ym, M 그룹, S 그룹 컬럼, shift_flag`
`data/processed/targets.parquet`: `dong_code, ym, visit_sh3, y_h1, y_h3, y_h6, established, resid_flag_h3, emerging_h3, emerging_h3_nores, valid`
`reports/08_typology.csv`: `dong_code, origin, pred_score, sig_visit, sig_fnb, sig_attn, type, realized_emerging, persistence_months`
`reports/05_leadlag_verdict.csv`: `series, segment, lp_fwd_sig_h, lp_rev_sig_h, best_lag_prewhitened, ccf_at_best, dh_p_x_to_visit, dh_p_visit_to_x, fwd_event_effect, rev_event_effect, verdict`
`reports/05_lp_irf.csv`: `shock, response, h, beta, se, ci_low, ci_high, n_obs`

---

## 14. 선후 관계 분석 정의 (Phase 5)

### 14.1 분석 대상 시계열

| 이름 | 컬럼 | 비고 |
|---|---|---|
| 방문 | `visit_g3` | 기준 시계열 |
| F&B 전체 | `net_rate_eb`, `fvi_d3` | |
| 독립 카페 | `indie_cafe_entry_rate` | 업종별 분리 |
| 프랜차이즈 | `franchise_entry_rate` | 업종별 분리 |
| 검색 | `search_g3` | |

모든 시계열은 동별로 차분 또는 성장률 형태로 쓰고, 월 횡단면 평균을 빼서 도시 공통 변동을 제거한다.

**정상성 확인**: 위 시계열 각각에 CIPS(기본)와 IPS 패널 단위근 검정을 하고, 단위근이 기각되지 않으면 한 번 더 차분한다. Pesaran CD 통계량도 함께 기록한다. 결과는 `reports/05_stationarity.md`.

**선후 분석에서는 3개월 평균(`_sh3`, `_g3`)이 아닌 월 단위 값을 쓴다.** 이동평균은 시점을 흐려 선후 판단을 왜곡하기 때문이다.
- 방문: `v_t = log(visit_share_t)`, 변화 `dv_t = v_t − v_{t−1}`
- 검색: `ds_t = log(search_index_t) − log(search_index_{t−1})`
- F&B: 월 단위 비율(`net_rate_eb`, 독립 카페·프랜차이즈 월 진입률의 EB 버전)

### 14.2 사전백색화 교차상관

1. x에 동 고정효과를 둔 패널 AR(p)를 적합한다: x_{i,t} = a_i + Σ_{l=1..p} φ_l x_{i,t−l} + u_{i,t}. p ≤ 3, AIC로 선택.
2. 같은 필터(a_i, φ)를 방문 변화 dv에 적용해 w_{i,t} = dv_{i,t} − a^v_i − Σ φ_l dv_{i,t−l}를 만든다(a^v_i는 dv의 동 평균).
3. ccf(k) = corr(u_{i,t}, w_{i,t+k}), k ∈ {−6, …, +6}. 동별로 계산해 동 평균한다.
4. 95% 신뢰구간: 동 단위 부트스트랩 1,000회.
5. k > 0 우세 → x 선행, k < 0 우세 → x 후행, k ≈ 0만 → 동시 변동.
6. 부록용으로 사전백색화 전 교차상관도 같은 방식으로 저장한다.

### 14.3 이벤트 정의

이벤트는 동마다 **처음 발생한 달 하나**만 쓴다. 기준선은 이벤트 직전 12개월의 동별 중앙값이다. 기준선 기간이 12개월 미만인 동은 이벤트에서 제외한다.

| 이벤트 | 정의 | 방향 | 반응 변수 |
|---|---|---|---|
| `ev_indie` | indie_cafe_entry_rate가 기준선의 2배 이상 AND 독립 카페 개업 3건 이상 | 정방향 | visit_sh3, search |
| `ev_fr` | franchise_entry_rate가 기준선의 2배 이상 AND 프랜차이즈 개업 2건 이상 | 정방향 | visit_sh3 |
| `ev_attn` | search_sh3가 기준선의 2배 이상 | 정방향 | visit_sh3, F&B 지표 |
| `ev_visit` | visit_sh3가 기준선의 1.3배 이상이고 다음 달에도 1.3배 이상 유지 | **역방향** | F&B 지표(개업률, 독립 카페·프랜차이즈 진입률, 순증률), search |

- 반응 경로는 이벤트 시점 기준 −12 ~ +12개월의 평균. 비교군은 같은 달에 이벤트가 없는 동(아직 이벤트 전이거나 이벤트가 없는 동)의 같은 기간 평균이다.
- 임계값(2배, 1.3배, 건수)은 민감도 분석 대상이다.
- 이벤트 시점에 `shift_flag == 1`이 끼면 그 이벤트는 제외한다.

### 14.4 방향성 판정 규칙

**국소 투영(§14.5)을 주 근거로** 하고, 사전백색화 교차상관·양방향 Granger·정·역방향 이벤트 스터디 중 **하나 이상**이 같은 방향일 때 `선행` 또는 `후행`으로 확정한다. 두 방향 충격반응이 모두 유의하면 `양방향`, 국소 투영과 나머지가 엇갈리거나 모두 약하면 `불명확`으로 둔다.

### 14.5 양방향 국소 투영 (Local Projections)

**정방향 (x 충격 → 방문 반응)**, h = 0, 1, …, 6:

```
v_{i,t+h} − v_{i,t−1} = α_i^h + γ_t^h + β_h · x_{i,t}
                        + Σ_{l=1..3} ρ_l^h · dv_{i,t−l} + Σ_{l=1..3} θ_l^h · x_{i,t−l} + ε_{i,t+h}
```

- 좌변은 t−1 대비 t+h까지의 방문 누적 변화. β_h는 "x가 1 표준편차 높을 때 h개월 뒤 방문이 얼마나 더 늘었나"의 충격반응이다.
- x는 학습 표본 전체에서 표준화(이 분석은 예측이 아닌 기술 분석이므로 전체 표본 사용 가능).
- α_i^h: 동 고정효과, γ_t^h: 월 고정효과(도시 공통 충격 흡수).
- 반응 변수와 충격 변수의 과거값을 통제해, 이미 진행 중이던 추세를 충격으로 오인하지 않게 한다.
- 표준오차: Driscoll-Kraay, 시차 h + 1 (겹치는 좌변과 횡단면 의존성에 강건).

**역방향 (방문 충격 → x 반응)**: 좌변을 x의 누적 변화(x가 비율이면 x_{i,t+h} − x_{i,t−1}), 충격을 dv_{i,t}로 바꿔 같은 식을 추정한다.

**산출물**: x별·방향별 β_h와 95% 신뢰구간 표(`reports/05_lp_irf.csv`: `shock, response, h, beta, se, ci_low, ci_high, n_obs`)와 충격반응 그림(정방향·역방향을 좌우로 나란히).

**주 결과 P2 판정**: x = 독립 카페 진입률일 때, 정방향 β_h의 95% 신뢰구간이 h = 1~6 중 하나 이상에서 0보다 크고, 역방향 반응이 유의하지 않거나 정방향보다 늦게 나타나면 "독립 카페 선행"으로 판정한다.
