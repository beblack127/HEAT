# 핫스팟 출현 예측 분석 계획 — Claude Code 작업 지시서 (v2, 기간·검증은 v3 `forecast_plan.md` 우선)

> Team #14 HEAT · 작성 2026-11-05 · v2: 인구이동을 타깃으로, F&B·검색량을 선행변수로 바꾼 버전
> 함께 보는 파일: `CLAUDE.md`(작업 규칙), `feature_spec.md`(지표·타깃 수식), `category_mapping.csv`(업종 매핑), `experiment_log_template.md`(실험 기록)
> 지표별 연구 근거 보고서: "F&B 상권 지표 정의와 연구 근거" 문서
>
> **2026-11-12 갱신: 기간·검증 구조·모델 목록은 `forecast_plan.md`(v3)가 이 문서보다 우선한다.** 학습 2022~2025, 시험 fold 2024·2025, 최종 시험 2026-01~09(한 번만), 예측 거리 3개월 기본·6·12개월 보조, 딥러닝 제외. 이 문서의 이동 데이터 타깃·선후 관계·유형 분류는 예측 작업 다음 단계로 남는다.

---

## 0. 한 줄 요약

**3개월 뒤 비거주 목적 방문이 지속적으로 늘어날 행정동을 예측하고, F&B 상권 변화와 검색량이 이동 데이터만 쓴 예측을 얼마나 개선하는지 검증한다.**
예측된 후보는 방문·F&B·검색 세 요소의 움직임으로 유형을 나눠 해석한다.

---

## 1. 설계

### 1.1 핵심 질문

"F&B 업체의 진입·생존·업종 변화, 검색량, 방문 증가 중 무엇이 먼저 움직이는가? 그 선후 관계를 이용해 아직 드러나지 않은 핫스팟 후보를 3개월 앞서 찾거나, 방문 증가가 지속될지 판별할 수 있는가?"

선후 관계는 가정하지 않고 양방향으로 검증한다. F&B가 방문을 이끌 수도(개척 업종이 수요를 만듦), 방문이 F&B를 이끌 수도(사업자가 늘어난 수요를 따라 들어옴), 단계에 따라 둘이 섞일 수도 있다. 어느 결과든 F&B 데이터의 역할이 달라질 뿐 의미가 있다(§5 판정표).

### 1.2 왜 이 구조인가

| 선택 | 이유 |
|---|---|
| 타깃 = 인구이동(비거주·비통근 방문) | "더 많은 사람이 온다"를 가장 직접 측정한다. 동 단위 일별 데이터라 품질이 가장 좋고, 귀가 도착으로 거주 효과를 분리할 수 있다 |
| F&B·검색량 = 선행변수 | 예측 대상과 독립된 데이터로 만든 피처여야 "추가 정보가 있다"를 검증할 수 있다. F&B 지수를 F&B 지표로 예측하는 순환 구조를 피한다 |
| 세 요소 동시 성장은 사후 해석에 사용 | 정의를 예측 타깃으로 쓰면 양성 사례가 너무 적고 검증이 순환적이 된다. 대신 예측된 후보의 성격을 분류하는 데 쓴다(§4 Phase 8) |

### 1.3 구조

```
[예측변수, origin t 시점까지]                     [타깃, t+3]
 M  이동 데이터 자체 (방문 시차, 귀가 도착 등)  ─┐
 F  F&B 지표 F01–F19 (인허가 + 추정매출)        ─┼─▶  비거주 방문 비중의 3개월 성장률
 S  검색량 (네이버 데이터랩 등, 팀원 제공)      ─┘      + 지속성 조건 → 이머징 핫스팟 라벨

[사후 해석] 예측 후보를 M·F·S의 최근 움직임으로 4개 유형으로 분류
```

### 1.4 가설

| ID | 가설 | 검증 방법 |
|---|---|---|
| H1 | 모델이 Naive(방문 변화 없음)보다 3개월 뒤를 잘 맞힌다 | MASE < 1, Precision@K |
| H2 | F&B 지표를 더하면 이동 데이터만 쓴 모델보다 예측이 좋아진다 | 실험 E1 vs E0, Diebold-Mariano 검정 |
| H3 | 거주 변화를 통제하면 가짜 핫스팟이 줄어든다 | 귀가 도착 피처·필터 유무에 따른 오탐 수 |
| H4 | 검색량을 더하면 예측이 좋아진다 | 실험 E2 vs E0, Diebold-Mariano 검정 |
| H5 | 세 요소가 함께 성장한 후보가 방문 성장을 가장 오래 유지한다 | 유형별 실현 지속률 비교 |
| HF1 | 순증률+회전율 조합이 순증률 단독보다 낫다 | F&B 내부 ablation |
| HF2 | F&B와 방문의 선후 관계는 업종·단계에 따라 다르다: 독립 카페 진입은 방문을 선행하고, 프랜차이즈 진입은 방문을 후행한다 | 양방향 국소 투영(주 방법), 사전백색화 교차상관(−6~+6), 양방향 Granger, 정·역방향 이벤트 스터디, 업종별 분리 |
| HF3 | 매출 지표를 더하면 과잉공급 동의 오탐이 줄어든다 | F&B 내부 ablation, 오탐 사례 |
| HF4 | 인접 동 F&B 변화가 추가 예측력을 준다 | F&B 내부 ablation |
| HS | 검색량과 방문의 선후 관계 (검색 선행을 예상) | 양방향 국소 투영, 사전백색화 교차상관, 양방향 Granger, 정·역방향 이벤트 스터디 |

### 1.5 주 결과 사전 확정 (pre-specified primary analyses)

실험이 20개 가까이 되면 우연히 좋아 보이는 결과가 섞인다. 그래서 **아래 세 비교만 주 결과로 미리 확정**하고, 나머지는 모두 탐색적 결과로 구분해 보고한다. 이 표는 E1을 처음 실행하기 전에 고정하며, 결과를 본 뒤 지표·임계값·비교 대상을 바꾸지 않는다. 바꿔야 하면 바꾼 이유와 원래 결과를 함께 보고한다.

| ID | 질문 | 비교 | 주 지표 | 판정 기준 |
|---|---|---|---|---|
| P1 | F&B가 방문 예측을 개선하는가 (H2) | LightGBM E1 vs E0, h = 3 | origin별 평균 MAE 차이 | Diebold-Mariano(HLN 보정) 단측 p < 0.05 |
| P2 | 독립 카페 진입이 방문을 선행하는가 (HF2) | 국소 투영: 독립 카페 진입 충격 → 방문 누적 반응, h = 1~6 | 반응 계수 β_h | h = 1~6 중 하나 이상에서 95% 신뢰구간이 0 위에 있고, 역방향(방문 → 독립 카페) 반응보다 먼저 나타남 |
| P3 | 동반 성장형이 가장 오래 가는가 (H5) | 동반 성장형 vs 방문 단독형 | 지속 개월 수 평균 | 동 단위 부트스트랩 95% 신뢰구간이 0을 포함하지 않음 |

- 세 주 결과에는 Holm 다중비교 보정을 적용한 p값도 함께 보고한다.
- 검색량(H4, HS)은 데이터 확보가 불확실하므로 주 결과에서 제외하고 탐색적 결과로 둔다.

### 1.6 역할 분담과 이 파일의 범위

- 이 계획은 F&B 담당 관점에서 전체 실험 파이프라인을 정의한다.
- 이동 데이터 집계(타깃·M 그룹)와 검색량 집계(S 그룹)는 팀원이 만든 월별 파일을 `data/external/`로 받아 쓴다. 스키마는 `feature_spec.md` §7(이동), §8(검색)에 정해 두었다.
- 검색량 파일이 아직 없으면 S 그룹 실험만 보류하고 나머지는 진행한다.

---

## 2. 입력 데이터

| 파일 | 내용 | 단위 | 주의 |
|---|---|---|---|
| `data/raw/` 일반음식점 인허가 (OA-23056) | 인허가일자, 영업상태, 폐업일자, 소재지면적, 주소, 사업장명, 업태구분명, 좌표 | 점포 | 현재 시점 스냅샷, 과거 폐업 점포 포함 |
| `data/raw/` 휴게음식점 인허가 (OA-16095) | 동일 구조 | 점포 | **편의점 포함** → 매핑표로 제외. 좌표계 EPSG:5174 |
| `data/raw/` 추정매출-행정동 | 분기 × 행정동 × 업종 매출, 주중·주말, 성별·연령대별 | 행정동×분기 | 분기 단위, 공표 시차 있음 |
| `data/external/mobility_monthly.csv` | 목적별 도착(귀가, 출근, 기타, 쇼핑, 관광 등), 주말 도착 | 행정동×월 | 팀 Dataset 1을 월 집계한 것. 데이터 수준 변동 구간 표시 필요 |
| `data/external/search_monthly.csv` (선택) | 동별 검색 지수 | 행정동×월 | 팀원 제공. 키워드 선정 방식 기록 필요 |
| `data/external/` 보조 이동 데이터 (선택) | 생활인구, 지하철 승하차 월 집계 | 행정동×월 | M 그룹 확장용 |
| `data/external/` 행정동 경계 (선택) | polygon | 행정동 | 인접 동 계산용 |

**Phase 0에서 가장 먼저 확인할 것**: 실제 컬럼명, 인코딩, 기간, 행정동 코드 체계(이동 425개, 생활인구 424개 차이 포함), `업태구분명`·`서비스_업종_코드` 고유값, 이동 데이터의 데이터 수준 변동 구간(제안서 EDA의 회색 구간) 위치. 결과는 `reports/00_data_audit.md`에 기록한다.

---

## 3. 폴더 구조

```
project/
├── CLAUDE.md
├── analysis_plan.md
├── feature_spec.md
├── category_mapping.csv
├── experiment_log_template.md
├── data/
│   ├── raw/              ← 인허가, 추정매출 원본 (수정 금지)
│   ├── external/         ← 이동·검색량 월 집계, 경계 파일
│   ├── interim/          ← 정제된 점포 테이블
│   └── processed/        ← 동-월 패널, 피처, 타깃 (parquet)
├── src/
│   ├── config.py
│   ├── 00_audit.py
│   ├── 01_clean_stores.py
│   ├── 02_build_panel.py
│   ├── 03_fnb_features.py
│   ├── 04_mobility_search_target.py
│   ├── 05_eda_leadlag.py
│   ├── 06_models.py
│   ├── 07_evaluate.py
│   └── 08_typology.py
├── experiments/
│   ├── log.csv
│   └── experiment_log.md
└── reports/
    └── figures/
```

---

## 4. 단계별 작업

### Phase 0. 데이터 감사 (`00_audit.py`)

모든 입력 파일의 행 수, 컬럼, 결측률, 날짜 범위, 코드 체계를 확인한다. 매핑표에 없는 업태구분명은 `reports/unmapped_categories.csv`로 뽑는다. 이동 데이터에서 데이터 수준 변동 월을 찾아 `config.py`의 `MOBILITY_EXCLUDED_MONTHS`에 기록한다.

### Phase 1. 점포 테이블 정제 (`01_clean_stores.py`)

1. 일반·휴게 데이터를 합치고 `source`로 구분한다.
2. `category_mapping.csv`로 `category`, `include`, `is_cafe`, `is_trend`를 붙이고 `include == 0`은 제외한다.
3. 행정동 매핑: 기존 EDA의 주소 기반 매핑을 재사용하고, 좌표가 있으면 경계 polygon으로 교차 검증한다.
4. 행정동 코드를 이동 데이터와 같은 기준 체계로 통일한다. **타깃과 피처의 키가 맞아야 하므로 이동 데이터의 코드 체계를 기준으로 삼는다.**
5. 일괄 폐업 탐지(`bulk_closure`): 같은 동·같은 폐업일 건수가 그 동 일평균의 10배 이상이면서 10건 이상.
6. 동일 위치 교체 탐지(`replaced`): 같은 좌표·주소에서 폐업일 ±30일 안에 새 인허가.
7. 프랜차이즈 판별(`is_franchise`): 정규화한 상호를 브랜드 목록과 매칭하거나, 서울 전역 동일 상호 N개 이상 규칙으로 대리한다.

산출물: `data/interim/stores_clean.parquet`

### Phase 2. F&B 동-월 패널 (`02_build_panel.py`)

월말 영업 점포 수 S_t, 개업 O_t, 폐업 C_t를 범주별로 계산한다. `bulk_closure` 유지(`v_keep`)·제외(`v_drop`) 두 버전을 만든다. 12개월 누적 지표를 위해 이동 데이터 시작보다 최소 12개월 앞선 2022년부터 만든다.

산출물: `data/processed/panel_base.parquet`

### Phase 3. F&B 지표 (`03_fnb_features.py`)

`feature_spec.md` §1~§5의 F01~F19와 §6의 FVI를 계산한다. 비율 지표는 `_ma3`와 `_eb`(Empirical Bayes 축소) 버전을 함께 만든다. 분기 매출은 공표 시차를 반영해 as-of로 월에 붙인다.

**F&B 활력지수(FVI)도 여기서 만든다.** 예측 타깃이 아니라 두 가지 용도의 요약 지표다.
- 모델 피처로 쓸 때: "F&B 지표 개별 투입 vs FVI 하나만 투입"을 비교해 요약 지수만으로 충분한지 본다.
- 사후 유형 분류(Phase 8)에서 F&B 축의 값으로 쓴다.

산출물: `data/processed/fnb_features.parquet`

### Phase 4. 이동·검색 피처와 타깃 (`04_mobility_search_target.py`)

`feature_spec.md` §7(이동), §8(검색), §9(타깃)대로 만든다.

- **타깃**: 비거주·비통근 목적 방문의 서울 전체 대비 비중을 3개월 평균한 뒤, 3개월 뒤까지의 로그 성장률을 구한다. 서울 비중을 쓰면 2025년 1월 같은 도시 전체 공통 변동이 빠진다. 3개월 평균은 일회성 급등을 누그러뜨린다.
- **이머징 라벨**: 현재 방문 상위 15%가 아니고, 3개월 성장률이 그 달 상위 10%이며, 이후 2개월도 성장이 유지되고, 귀가 도착이 급증한 거주형이 아닌 동.
- **M 그룹 피처**: 방문 비중 성장률 시차, 방문 수준, 귀가 도착 성장률(거주 통제), 주말 방문 비중, (있으면) 생활인구·지하철.
- **S 그룹 피처**: 검색 지수 성장률, 시차, 이동평균.
- 데이터 수준 변동 월이 계산 창에 들어가는 관측은 결측으로 두고 학습·평가에서 뺀다. 보간하지 않는다.

산출물: `data/processed/mobility_search_features.parquet`, `data/processed/targets.parquet`

### Phase 5. EDA와 선행-후행 분석 (`05_eda_leadlag.py`)

0. **사전 점검: 정상성과 횡단면 의존성** — 선후 분석(2~5번)의 전제를 먼저 확인한다.
   - 분석에 쓰는 모든 시계열(`feature_spec.md` §14.1)에 패널 단위근 검정을 한다. 기본은 횡단면 의존성을 고려하는 CIPS(Pesaran 2007), 비교용으로 IPS(Im, Pesaran & Shin 2003). 단위근이 기각되지 않으면 한 번 더 차분하고, 그 사실을 기록한다.
   - Pesaran CD 검정으로 동 간 상관의 크기를 보고한다. 월 횡단면 평균을 뺀 뒤에도 CD가 크면, 이후 표준오차는 횡단면 의존성에 강건한 방식(Driscoll-Kraay)을 쓴다.
   - 결과는 `reports/05_stationarity.md`에 표로 남긴다.
1. **사례 동 추이**: 성수, 연남, 한남, 신사, 둔촌1동, 삼성1동, 역삼2동에서 방문 성장, FVI, 주요 F&B 지표, 검색량을 한 그림에 겹쳐 그린다. 어느 것이 먼저 움직였는지를 눈으로 확인한다. 둔촌1동은 방문 증가가 귀가·생활형 업종과 함께 나타나 핫스팟과 구분되는지 본다(H3).
2. **사전백색화 후 양방향 교차상관 (−6 ~ +6개월)**: 자기상관이 강한 두 시계열은 관계가 없어도 교차상관이 크게 나오므로, 반드시 사전백색화(prewhitening, Box-Jenkins) 후 계산한다.
   - 절차: 설명 쪽 시계열 x에 동 고정효과를 둔 패널 AR(p)(p ≤ 3, AIC로 선택)를 적합하고, 같은 필터를 방문 시계열에도 적용한 뒤 두 잔차의 교차상관을 구한다(`feature_spec.md` §14.2).
   - 시차 k는 "x의 t월 값 vs 방문의 t+k월 값". k > 0 우세면 x 선행, k < 0 우세면 x 후행, k = 0 근처만 크면 동시 변동(공통 원인 가능성).
   - 동 평균 교차상관의 95% 신뢰구간은 동 단위 부트스트랩(1,000회)으로 구한다. 동별 최대 상관 시차의 분포도 함께 그린다.
   - 비교를 위해 사전백색화 전 교차상관도 부록에 남긴다. 둘이 크게 다르면 그 차이 자체를 보고한다.
3. **패널 Granger 검정 (양방향)**: Dumitrescu & Hurlin (2012) 검정으로 F&B→방문, 방문→F&B, 검색→방문, 방문→검색을 각각 검정한다. 이 검정은 T가 N보다 훨씬 작으면 크기 왜곡이 생길 수 있으므로(Juodis, Karavias & Sarafidis 2021) 보조 증거로만 쓴다. Granger는 예측적 선행이지 인과가 아니라고 명시한다.
3-1. **양방향 국소 투영 (Local Projections) — 선후 관계의 주 방법**: Jordà (2005)의 방법을 패널 고정효과로 확장해, "x 충격 후 h개월 동안 방문이 어떻게 반응하는가"를 시점별로 직접 추정한다(식은 `feature_spec.md` §14.5).
   - h = 0 ~ 6, 동·월 고정효과, 반응 변수와 충격 변수의 시차 3개를 통제한다. 표준오차는 Driscoll-Kraay(시차 h+1).
   - 방향을 바꿔(방문 충격 → x 반응) 똑같이 추정한다. 두 충격반응 그림을 나란히 놓으면 선후 관계가 한눈에 보인다.
   - 독립 카페, 프랜차이즈, F&B 전체(순증률), 검색량 각각에 대해 수행한다.
   - 장점: 이벤트 스터디보다 통제변수를 넣기 쉽고, 임계값 선택에 의존하지 않으며, 신뢰구간이 있는 충격반응 함수를 준다. 주 결과 P2는 이 결과로 판정한다.
4. **정·역방향 이벤트 스터디**: 이벤트 정의는 `feature_spec.md` §14를 따른다.
   - 정방향: F&B 이벤트(독립 카페 진입 급증), 관심 이벤트(검색 급증) 기준으로 전후 ±12개월 **방문 비중**의 평균 경로를 비이벤트 동과 비교한다.
   - 역방향: 방문 이벤트(방문 비중 급증) 기준으로 전후 ±12개월 **F&B 지표**(개업률, 독립 카페 진입률, 프랜차이즈 진입률, 순증률)와 **검색 지수**의 평균 경로를 비교한다.
   - 이벤트 이전 구간(−12 ~ −1)에 이미 상대 지표가 움직였는지가 선후 판단의 핵심이다.
5. **업종별 분리 검증**: 2~4번(3-1 포함)을 독립 카페 진입(`indie_cafe_entry_rate`)과 프랜차이즈 진입(`franchise_entry_rate`)에 대해 따로 수행한다. "독립 카페는 선행, 프랜차이즈는 후행"이 나오면 젠트리피케이션 단계론(초기 개척 업종 유입 → 방문 증가 → 프랜차이즈 진입)과 일치한다. 가능하면 성장 단계(방문 상위 확립 전/후)로도 나눠 본다.
6. **공급-수요 유형 전이행렬**: F17 유형의 분기 간 전이 확률과, 유형별 다음 분기 방문 성장의 평균.
7. **방향성 판정**: 2~5번을 종합해 F&B 지표별·업종별로 아래 셋 중 하나로 판정하고 `reports/05_eda_leadlag.md`에 표로 남긴다. **국소 투영을 주 근거로 하고**, 나머지(사전백색화 교차상관, Granger, 이벤트 스터디) 중 하나 이상이 같은 방향일 때 확정한다. 엇갈리면 `불명확`으로 둔다.

| 판정 | 증거 패턴 | F&B 데이터의 역할 |
|---|---|---|
| 선행 | x 충격 → 방문 반응이 h ≥ 1에서 유의, 역방향 반응은 없거나 늦음, 사전백색화 교차상관 k > 0 우세 | **조기 경보 신호**: 예측변수로서 E1 > E0 기대 |
| 후행 | 방문 충격 → x 반응이 유의, 정방향 반응은 없거나 늦음, k < 0 우세 | **지속성 확인 신호**: 방문이 늘었는데 F&B가 따라오지 않으면 일회성 의심. 이머징 판별과 유형 분류(H5)에 활용 |
| 양방향 | 두 방향 충격반응이 모두 유의 | **상호 강화 순환**: 동반 성장형이 가장 오래 갈 것(H5) |

산출물: `reports/05_stationarity.md`, `reports/05_eda_leadlag.md`, `reports/05_leadlag_verdict.csv`, `reports/figures/`(충격반응 그림, 교차상관 그림, 이벤트 스터디 그림)

### Phase 6. 모델링 (`06_models.py`)

#### 6.1 문제 설정

- **회귀**: origin t에서 y = 방문 비중 3개월 로그 성장률(t → t+3)을 예측한다. h = 3이 기본이고 1, 6은 확장이다.
- **랭킹**: origin t에서 이머징 라벨 확률을 예측해 상위 K개를 고른다.
- **direct 방식**: horizon별로 모델을 따로 학습한다.

#### 6.2 처음 시작할 모델 (추천)

| 순서 | 모델 | 왜 | 구현 |
|---|---|---|---|
| 1 | Naive(성장률 0 = 방문 비중 유지), 모멘텀(최근 3개월 성장률 그대로), Seasonal Naive | 제안서 H1의 기준선 | pandas |
| 1-1 | **전역 Ridge 회귀** — 강한 선형 기준선 | LightGBM과 같은 피처로 모든 동을 함께 학습하는 선형 모델. LightGBM의 개선이 비선형성 덕분인지, 동을 함께 학습한 덕분인지 구분한다. Ridge ≈ LightGBM이면 단순한 Ridge를 주 모델로 보고하는 것이 낫다 | scikit-learn Ridge. 표준화는 학습 구간에서만, alpha는 학습 구간 내 시계열 분할로 선택 |
| 2 | **전역 LightGBM (회귀 + 분류)** — 주력 | 동 425개 × 월 약 36개로 동별 시계열은 짧고 동 수는 많다. 모든 동을 한 모델로 학습하는 cross-learning이 유리하다(M5 대회 상위 방법 모두 cross-learning, 우승 방법은 LightGBM 기반). 피처 그룹 추가 실험을 같은 틀에서 할 수 있다 | lightgbm |
| 3 | **이원고정효과 선형회귀** — 해석용 | 동·월 고정효과로 동 고유 수준과 도시 공통 충격을 흡수한 뒤, F&B·검색 지표의 효과를 계수로 읽는다 | linearmodels PanelOLS. 표준오차는 Driscoll-Kraay(횡단면 의존성과 3개월 겹침에 따른 자기상관에 강건), 동·월 이중 군집을 강건성 확인으로 병기 |
| 4 | 로지스틱 회귀(L2) — 랭킹 해석용 | 골목상권 성장 요인 연구 방식과 비교 가능(강현모·이상경 2019) | scikit-learn |
| 5 | 동별 ARIMA / ARIMAX | 제안서 비교 대상. 시점이 짧아 12개월 계절 차분은 쓰지 않고, 외생변수는 2~3개로 제한 | statsforecast, statsmodels |

**LightGBM 시작 설정**: `num_leaves=15, min_data_in_leaf=50, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=2000 + early stopping(학습 구간 마지막 3개월)`. 분류는 `is_unbalance=True`.

#### 6.3 실험 설계

**주 실험 (데이터 소스별 추가 가치)** — 같은 모델, 같은 origin에서 피처 그룹만 바꾼다.

| 실험 | 피처 | 검증 |
|---|---|---|
| E0 | M (이동 데이터만) | 기준 모델. Ridge와 LightGBM 모두 실행 |
| E1 | M + F (F&B 지표 개별) | H2 |
| E1-fvi | M + FVI 하나 | 요약 지수로 충분한가 |
| E2 | M + S (검색량) | H4 |
| E3 | M + F + S | 전체 |
| E-F | F만 (이동 없이) | F&B 단독 정보량 |
| E0-nores | M에서 귀가 도착 피처 제외 + 라벨의 거주 필터 제외 | H3 |

**F&B 내부 실험** — E1을 기준으로 F&B 그룹을 하나씩 더한다.

| 실험 | 피처 | 검증 |
|---|---|---|
| F1' | M + F04 순증률만 | HF1 비교 기준 |
| F1 | M + G1 재고·흐름 | HF1 |
| F2 | F1 + G2 업력·생존 | |
| F3 | F2 + G3 업종 구성 | |
| F4 | F3 + G4 공간 | HF4 |
| F5 | F4 + G5 매출 (= E1) | HF3 |

### Phase 7. 평가 (`07_evaluate.py`)

**검증 방식**: 확장 창 rolling-origin. 매월 origin을 한 칸씩 옮기며 origin 이전 데이터로만 학습한다(Hyndman & Athanasopoulos, FPP).

**검증 구조**: `forecast_plan.md` §2를 따른다 (v3로 대체).
- fold F2024: 학습 2022-01~2023-12, 시험 2024년. fold F2025: 학습 2022-01~2024-12, 시험 2025년. 2023년 시험은 하지 않는다.
- 최종: 학습 2022-01~2025-12, 시험 2026-01~09. 모든 선택을 고정한 뒤 한 번만 실행한다.
- 결과는 fold별로 함께 보고하고, 평균 하나로 결론 내리지 않는다.

**지표**

| 과제 | 지표 | 해석 |
|---|---|---|
| 회귀 | MAE, RMSE, MASE(Naive 대비) | MASE < 1이면 Naive보다 낫다 |
| 랭킹 | Precision@K, Recall@K (K = 10, 20, 30) | 상위 K개 추천 중 실제 이머징 비율 |
| 추가 가치 | E1−E0, E2−E0의 origin별 차이와 부호 일치 비율 | 대부분 origin에서 개선되는가 |
| 유의성 | Diebold-Mariano 검정(HLN 소표본 보정), 랭킹은 origin 단위 부트스트랩 | 차이가 우연이 아닌가 (아래) |
| 진단 | 잔차 Ljung-Box(ARIMA), 오탐·미탐 사례 | 남은 자기상관, 왜 틀렸나 |

**누수 점검 체크리스트** (매 실행 `experiments/log.csv`에 기록)
- [ ] t 시점 피처가 t월 말 이후 정보를 쓰지 않는다
- [ ] 생존율은 관측이 끝난 코호트만 쓴다
- [ ] 분기 매출은 공표 시차 이후 월에만 쓴다
- [ ] 타깃 분위 임계값은 각 origin의 횡단면에서만 계산한다
- [ ] 스케일링·EB 사전분포·하이퍼파라미터는 학습 구간에서만 정한다

**예측 성능 차이의 유의성 검정**

두 모델 A(예: E0), B(예: E1)의 차이가 우연인지 확인한다. MAE가 작다는 것만으로는 H2·H4를 주장할 수 없다.

1. **손실 차이 시계열 만들기**: origin t마다 동 평균 손실 차이 d_t = mean_i(|e^A_{i,t}|) − mean_i(|e^B_{i,t}|)를 구한다. d_t > 0이면 B가 낫다.
2. **Diebold-Mariano 통계량**: DM = mean(d) / sqrt(LRV(d) / n). h = 3 예측을 매월 하면 예측 구간이 겹쳐 d_t에 MA(2) 형태의 자기상관이 생기므로, 장기분산 LRV는 시차 h − 1 = 2의 Newey-West(HAC)로 추정한다(Diebold & Mariano 1995).
3. **소표본 보정**: origin이 10개 안팎이므로 Harvey, Leybourne & Newbold (1997)의 보정 계수를 곱하고 자유도 n − 1의 t분포로 p값을 구한다.
4. **강건성 확인**: 동 × origin 수준의 손실 차이를 상수에 회귀하고 Driscoll-Kraay 표준오차로 검정한 결과도 함께 보고한다.
5. **랭킹 비교**: Precision@K 차이는 origin을 단위로 재표본하는 부트스트랩(2,000회)으로 95% 신뢰구간을 구한다.
6. **주 결과 P1**에는 Holm 보정 p값을 함께 적는다(§1.5).

### Phase 8. 사후 유형 분류 (`08_typology.py`)

예측된 후보(그리고 실제 이머징 동)를 origin 시점의 세 축 움직임으로 분류한다. 제안서의 핫스팟 정의(관심·방문·F&B 집적이 함께 지속 성장)를 여기서 반영한다.

| 축 | 신호 (origin t 기준, 최근 6개월) |
|---|---|
| 방문 | 방문 비중 6개월 성장률이 그 달 상위 1/3 |
| F&B | FVI 6개월 변화가 그 달 상위 1/3 |
| 관심 | 검색 지수 6개월 성장률이 그 달 상위 1/3 |

| 유형 | 조건 | 해석 |
|---|---|---|
| 동반 성장형 | F&B 신호와 관심 신호 모두 | 정의에 가장 부합. 지속 가능성 가장 높을 것으로 예상(H5) |
| F&B 주도형 | F&B 신호만 | 업체가 먼저 들어온 초기 단계 |
| 관심 주도형 | 관심 신호만 | 화제성 선행. 일회성 위험 점검 |
| 방문 단독형 | 둘 다 없음 | 거주 증가·이벤트성 의심 |

**H5 검증**: 유형별로 실제 방문 성장이 t+3 이후 몇 개월 유지됐는지(지속률)를 비교한다.
검색량이 없으면 F&B 축만으로 "F&B 동반형 / 방문 단독형" 2유형으로 분류한다.

산출물: `reports/08_typology.md`, 후보 동 지도와 유형별 지속률 그림

---

## 5. 해석 가이드

| 관찰 패턴 | 해석 | 근거 |
|---|---|---|
| E1이 E0보다 대부분 origin에서 낫다 | F&B 데이터가 이동 데이터에 없는 선행 정보를 가진다 | Glaeser, Kim & Luca 2018 |
| E1 ≈ E0인데 Phase 5 판정이 "후행" | F&B는 조기 신호가 아니라 확인 신호다. 이머징 라벨의 지속성 판별, 유형 분류에서 기여를 따로 평가한다 | |
| 독립 카페 선행 + 프랜차이즈 후행 | 단계별로 역할이 다르다. 초기 탐지에는 독립 카페, 성숙 판정에는 프랜차이즈 | 류화연·박진아; 구형모 |
| 동시 변동만 뚜렷하다 | 미디어·교통 개통 같은 공통 원인 가능성. 사례 동에서 외부 이벤트를 확인 | |
| Ridge ≈ LightGBM | 개선은 비선형성보다 여러 동을 함께 학습한 덕분이다. 단순한 Ridge를 주 모델로 보고하는 것이 낫다 | |
| E1 < E0이지만 DM 검정 비유의 | 개선 방향은 맞지만 이 기간·origin 수로는 우연과 구분되지 않는다. "근거 불충분"으로 보고 | Diebold & Mariano 1995 |
| E1-fvi ≈ E1 | 요약 지수 하나로 충분하다 → 팀 Hotness Score에 FVI를 성분으로 넣을 근거 | |
| F&B 내부 실험에서 G3(업종 구성)의 기여가 크다 | 무엇이 들어오는지가 얼마나 들어오는지보다 중요하다 | 구형모; Behrens et al. 2024 |
| 순증률↑ + 점포당 매출↓ 동이 오탐에 많다 | 과잉 공급 신호를 필터로 써야 한다 | 공급-수요 2×2 |
| 귀가 피처 제거 시 둔촌1동형 오탐 증가 | 거주 통제가 필요하다(H3) | 제안서 EDA |
| 동반 성장형의 지속률이 가장 높다 | 세 요소 동시 성장 정의의 타당성 지지(H5) | |
| 개선이 없다 | 이 기간·해상도에서는 F&B가 방문을 선행하지 않는다. 그대로 보고한다 | |

**보고서에 넣을 결과물**
0. 주 결과 P1~P3 요약표 (사전 확정 기준, Holm 보정 p값 포함)
1. 모델별 MAE·MASE·Precision@K (Naive, Ridge 대비)
2. E0~E3 비교 막대그래프, origin별 개선 분포, Diebold-Mariano 검정표
3. F&B 내부 ablation 그래프
4. LightGBM SHAP 요약도, 이원고정효과 계수표(Driscoll-Kraay 표준오차)
5. 선후 관계 결과 (HF2, HS): 정상성·CD 검정표, **양방향 충격반응 그림(국소 투영)**, 사전백색화 교차상관 그림, 양방향 Granger 표, 정·역방향 이벤트 스터디 그림, 업종별 방향성 판정표
6. 2026년 최종 시험 예측 상위 20개 동 지도, 유형 표시, 적중·오탐 사례
7. 유형별 지속률 비교 (H5)

---

## 6. 위험과 대응

| 위험 | 대응 |
|---|---|
| 평가 origin이 10개 안팎으로 적다 | origin별 결과 공개, 부호 일치 비율 보고, DM 검정에 HLN 소표본 보정, 과한 튜닝 금지 |
| 자기상관 때문에 교차상관이 가짜로 커진다 | 사전백색화 후 계산, 국소 투영을 주 방법으로 |
| 3개월 예측의 겹침으로 오차가 자기상관된다 | DM 검정에 시차 2 HAC, 회귀 표준오차는 Driscoll-Kraay |
| 동 간 공통 충격으로 표본이 실제보다 작다 | Pesaran CD 보고, 횡단면 의존성에 강건한 검정·표준오차 사용 |
| 실험이 많아 우연한 개선이 섞인다 | 주 결과 3개 사전 확정(§1.5), Holm 보정, 나머지는 탐색적 결과로 표시 |
| 이동 데이터 수준 변동 구간 | 해당 월이 낀 관측 제외, 서울 비중으로 정규화 |
| 이머징 양성 사례가 적다 | 임계값 민감도(상위 10·15·20%), 회귀 결과를 주 결과로 |
| 작은 수로 F&B 비율이 출렁인다 | EB 축소, 3개월 이동합, winsorize |
| 일괄 폐업, 양수·양도 | bulk_closure 민감도 패널, replaced 플래그 |
| 매출의 분기 단위·공표 시차 | as-of 결합, 시차 파라미터화 |
| 검색량을 동 단위로 만들기 어렵다 | 키워드 규칙 문서화, 매칭 실패 동은 결측. S 그룹 실험은 검색량이 있는 동만으로 별도 보고 |
| 행정동 체계 불일치 | 이동 데이터 코드 체계로 통일 |

---

## 7. 진행 순서 (Claude Code에 그대로 요청할 단위)

1. "Phase 0 데이터 감사를 실행하고 reports/00_data_audit.md를 만들어줘. 이동 데이터의 데이터 수준 변동 월도 찾아줘."
2. "category_mapping.csv를 감사 결과로 보완하고 Phase 1 정제를 실행해줘. 행정동 코드는 이동 데이터 기준으로 맞춰줘."
3. "Phase 2 패널과 Phase 3 F&B 지표, FVI를 feature_spec.md대로 계산해줘."
4. "Phase 4 이동·검색 피처와 타깃, 이머징 라벨을 만들어줘. 양성 비율도 보고해줘."
5. "Phase 5를 실행해줘. 먼저 CIPS·IPS 단위근 검정과 Pesaran CD 검정을 하고, 양방향 국소 투영 충격반응, 사전백색화 교차상관(−6~+6), 양방향 Granger, 정·역방향 이벤트 스터디를 독립 카페와 프랜차이즈를 나눠서 해줘. 마지막에 방향성 판정표를 만들어줘."
6. "베이스라인, E0 Ridge, E0 LightGBM, 이원고정효과 회귀(Driscoll-Kraay)를 rolling-origin으로 평가하고 log.csv에 기록해줘. 이 시점에 analysis_plan §1.5의 주 결과 표를 그대로 고정해줘."
7. "E1~E3, E1-fvi, E-F, E0-nores와 F&B 내부 실험 F1'~F5를 돌리고, E0 대비 Diebold-Mariano 검정(HLN 보정)과 랭킹 부트스트랩 신뢰구간까지 포함해 결과 표와 그림을 만들어줘. 주 결과 P1~P3는 따로 요약해줘."
8. "Phase 8 유형 분류와 유형별 지속률 비교를 해줘."
9. "모든 선택을 고정하고 최종 시험(2026-01~09, `forecast_plan.md` §2)을 한 번 실행해줘."

---

## 8. 참고문헌

**지표 근거**
- 최은준·천상현·이수기 (2021). 사업체의 생존·폐업 기간을 활용한 서울시 상업공간의 변화분석. 지역연구 37(4). https://repository.hanyang.ac.kr/handle/20.500.11754/177181
- 천상현·김지은 (2022). 상업젠트리피케이션에 따른 기존 상인의 이탈. https://koreascience.kr/article/JAKO202221533775587.page
- 구형모. 서울 상업 젠트리피케이션 진행 단계의 시·공간 분포 탐색: 커피전문점의 밀도 변화를 중심으로. https://m.earticle.net/Article/A375860
- 김경선·김동섭 (2019). 서울시 상업젠트리피케이션 영향요인에 관한 연구. https://koreascience.kr/article/JAKO201909469054707.pdf
- 강현모·이상경 (2019). 시계열 군집분석과 로지스틱 회귀분석을 이용한 골목상권 성장요인 연구. https://koreascience.kr/article/JAKO201912062687876.pdf
- 서울시 홍대 주변 상권의 공간적 변화에 관한 연구 (2022). https://scholar.kyobobook.co.kr/article/detail/4010036832807
- Glaeser, Kim & Luca (2018). Nowcasting Gentrification. AEA P&P 108. https://hbswk.hbs.edu/item/nowcasting-gentrification-using-yelp-data-to-quantify-neighborhood-change
- Glaeser, Luca & Moszkowski (2023). Gentrification and Retail Churn. RSUE 100. https://emoszkowski.github.io/ericamoszkowski.com/GentrificationAndRetailChurn_vF.pdf
- Behrens, Boualam, Martin & Mayneris (2024). Gentrification and Pioneer Businesses. REStat 106(1). https://econpapers.repec.org/RePEc:cpr:ceprdp:13296
- Sung, Lee & Cheon (2015). Operationalizing Jane Jacobs's Urban Design Theory. JPER 35(2). https://architexturez.net/node/168025
- Cho et al. (2023). Survival characteristics of F&B businesses in a gentrified commercial area. Heliyon 9(11). https://doaj.org/article/5478247e3bbd4618bf107ad2f799f8ed

**방법론 근거**
- Hyndman & Athanasopoulos. Forecasting: Principles and Practice (rolling forecasting origin). https://otexts.com/fpp3/
- M5 대회 우승 방법(LightGBM, 다층 데이터 풀링). https://dspace.kaist.ac.kr/handle/10203/299580
- Januschowski et al. Forecasting with Trees. https://cdn.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf
- Dumitrescu & Hurlin (2012). Testing for Granger non-causality in heterogeneous panels. Economic Modelling 29(4). https://econpapers.repec.org/RePEc:eee:ecmode:v:29:y:2012:i:4:p:1450-1460
- Juodis, Karavias & Sarafidis (2021). A homogeneous approach to testing for Granger non-causality in heterogeneous panels. Empirical Economics. https://bura.brunel.ac.uk/bitstream/2438/30715/1/FullText.pdf
- Clayton & Kaldor (1987). Empirical Bayes estimates for disease mapping. Biometrics 43(3). https://search.r-project.org/CRAN/refmans/DCluster/html/lognormalEB.html
- Lim, Arik, Loeff & Pfister (2021). Temporal Fusion Transformers. IJF. https://arxiv.org/abs/1912.09363
- Box & Jenkins. Time Series Analysis: Forecasting and Control (사전백색화 후 교차상관 식별)
- Jordà (2005). Estimation and Inference of Impulse Responses by Local Projections. American Economic Review 95(1), 161–182.
- Diebold & Mariano (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics 13(3), 253–263.
- Harvey, Leybourne & Newbold (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting 13(2), 281–291.
- Im, Pesaran & Shin (2003). Testing for unit roots in heterogeneous panels. Journal of Econometrics 115(1), 53–74.
- Pesaran (2007). A simple panel unit root test in the presence of cross-section dependence. Journal of Applied Econometrics 22(2), 265–312.
- Pesaran (2021). General diagnostic tests for cross-sectional dependence in panels. Empirical Economics 60, 13–50. (원 논문 2004 working paper)
- Driscoll & Kraay (1998). Consistent covariance matrix estimation with spatially dependent panel data. Review of Economics and Statistics 80(4), 549–560.

**데이터**
- 일반음식점 인허가 OA-23056: https://data.seoul.go.kr/dataList/OA-23056/S/1/datasetView.do
- 휴게음식점 인허가 OA-16095: https://data.seoul.go.kr/dataList/OA-16095/S/1/datasetView.do
- 상권분석서비스 제공정보: https://golmok.seoul.go.kr/images/seoul_info.pdf
- 추정매출(분기) 메타데이터: https://www.data.go.kr/biz/dcat/metadata/15147229.do
- 수도권 생활이동 (팀 Dataset 1): https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do
