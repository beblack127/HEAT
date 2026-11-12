# CLAUDE.md — HEAT (F&B·검색 파트)

한양대학교 시계열 수업 Team #14 HEAT. 서울의 현재 핫스팟과 떠오르는 핫스팟을 찾는 시스템을 만든다.
이 저장소에서 관리하는 것은 **행정동 이름표, 네이버 검색량, 음식점 개폐업** 세 가지다.

지금 할 일: 2022~2025년으로 학습해 2026년의 **행정동별 검색량**과 **개폐업**을 예측한다 (`docs/forecast_plan.md`).

## 먼저 읽을 파일

1. `docs/forecast_plan.md` — **v3 예측 계획. 다른 문서와 다르면 이 문서가 우선**
2. `docs/feature_spec.md` — F&B 지표 수식과 시점 규칙. 코드는 이 명세를 따른다
3. `config/category_mapping.csv` — 업태구분명 → 분석 범주, 포함/제외
4. `reports/nametags/00_결과_요약.md`, `reports/search/00_검색량_요약.md` — 이름표·검색량을 어떻게 만들었는지
5. `docs/analysis_plan.md` — v2 설계(이동 타깃, 선후 관계, 유형 분류). 예측 다음 단계용. 기간·검증은 v3를 따른다

## 저장소 구조

```
config/                 category_mapping.csv
data/
  raw/                  (git 제외) 인허가 원본 등. 아래 '로컬에만 있는 파일' 참고
  raw/search_api/       네이버 API HUB 응답 원본 (batch_XXXX.json, pilot.json)
  external/geo/         행정동·법정동 경계, 한강, 상권 영역, 역 위치
  interim/              (git 제외) 음식점_정리본.csv, 음식점_개폐업사건.csv
  processed/nametags/   이름표 01~07 CSV, 지도용 JSON
  processed/search/     검색량 묶음·조각·동별 지수
  processed/fnb/        음식점_월별집계_잠정.csv (서울 전체 월 집계)
maps/                   단독 HTML 지도 (검색량+이름표, 이름표 확인용)
src/nametags/           이름표 만들기, 지도 도형, 지도 템플릿
src/search/             검색량 수집·동별 지수·지도
src/preprocess/         인허가 전처리
src/eda/                초기 EDA, 중간발표 그림
src/forecast/           (이제 만들 것) 예측 파이프라인
reports/                사람용 결과 (md, 그림)
experiments/            실험 기록
```

### 로컬에만 있는 파일 (git에 올리지 않음, 100MB 이상이거나 원본)

- `data/interim/음식점_정리본.csv` (≈285MB), `data/interim/음식점_개폐업사건.csv` (≈113MB): `src/preprocess/preprocess_restaurant_data.py` 산출물
- `data/raw/` 인허가 원본 (일반음식점 OA-23056, 휴게음식점 OA-16095), 상권분석서비스 원본
- 팀 공유 폴더의 같은 이름 파일을 이 위치에 두고 작업한다. 파일이 없으면 사용자에게 먼저 묻는다

## 비밀 정보

- 네이버 API 키는 `.env`(`NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`)에만 둔다. `.env`, `*.rtf`, 키가 든 파일은 **절대 커밋하지 않는다** (`.gitignore`에 있음)
- 키 값을 출력하거나 로그·보고서·코드에 적지 않는다
- 검색량 수집은 끝났다. 다시 받을 때만 `src/search/collect_datalab.py`를 쓴다 (이미 받은 요청은 건너뜀)

## 반드시 지킬 규칙

### 기간과 검증 (v3)
- 학습 2022-01~2025-12. 시험 fold F2024(학습 ~2023-12), F2025(학습 ~2024-12). 2023년 시험은 하지 않는다
- 최종 시험 2026-01~09는 모든 선택을 고정하고 커밋한 뒤 **한 번만** 실행한다
- 예측 거리 3개월 기본, 6·12개월 보조. direct multi-horizon
- 2026-10은 부분 월이라 시험에 넣지 않는다
- 무작위 K-fold는 쓰지 않는다

### 시점과 누수
- origin t의 입력에는 t월 말까지 알 수 있던 값만 쓴다
- 표준화, EB 사전분포, 하이퍼파라미터, 계절 계수는 fold 학습 구간에서만 정한다
- **검색량 `seasonal_factor`·`chg_3m_sa`는 CSV에 2022~2024 기준으로 들어 있다. 모델링에서는 fold마다 다시 계산한다**
- 미래 값이 필요한 외생변수는 시차 ≥ h로만 넣는다 (코로나 더미처럼 미래가 정해진 변수는 예외)
- 생존율은 관측이 끝난 코호트만 쓴다 (feature_spec F09)

### 검색량
- 동 비교는 절대 지수가 아니라 **서울 대비 비중**(`seoul_share`)으로 한다
- 제외 규칙(2022년 월평균 < 1, 0인 달 6개월 이상, 서울 밖 시·군 이름, 검토 목록 11개)은 고정. 모델 결과를 본 뒤 목록을 바꾸지 않는다
- 상대 지수다. "몇 건"이라고 쓰지 않는다

### 개폐업
- `category_mapping.csv`의 `include == 0` 업종(편의점, 구내매장 등)은 제외한다
- 매핑표에 없는 업태구분명은 추측하지 말고 `reports/unmapped_categories.csv`에 모아 사용자에게 보고한다
- 일괄 폐업(`bulk_closure`)은 플래그로 남기고 `v_keep`·`v_drop` 두 패널로 모두 돌린다
- 폐업 신고 지연을 모델링 전에 점검한다
- 코로나 더미: 2022-01~04
- 인허가 좌표는 섞여 있다: x 150k~260k·y 400k~500k는 EPSG:5174, x > 800k·y > 1.8M은 EPSG:5179
- 비율의 분모가 0이면 NaN. 0으로 채우지 않는다

### 이름표
- 동네 칸: 법정동 면적이 행정동 육지 면적의 10% 이상. 역 칸: 같은 강변 1000m 원 안 육지 면적 5% 이상. 상권 칸: 상권 면적의 10% 이상이 동 안
- 육지 면적 = 행정동 − 한강 폴리곤. 자세한 규칙은 `reports/nametags/00_결과_요약.md`

### 모델과 평가
- 모델: Naive, Seasonal Naive, ETS, AutoARIMA(SARIMA), Theta, Croston/TSB, SARIMAX, LightGBM(개수는 Poisson), Ridge. 딥러닝은 쓰지 않는다
- SARIMA·SARIMAX는 검색량 모든 동, 개폐업은 서울·구·개폐업 많은 동에만
- 모든 모델을 Naive·Seasonal Naive와 비교한다. 못 넘으면 그대로 보고한다
- LightGBM과 같은 피처의 Ridge를 항상 함께 돌린다
- 지표: MASE(계절 naive 분모), Poisson deviance, top-K 적중률(K = 10, 20, 30), DM 검정(HAC 시차 h − 1, HLN 보정)
- MAE 차이만으로 "개선"이라고 쓰지 않는다. fold별 결과를 함께 보고한다
- 실행할 때마다 `experiments/log.csv`에 한 줄을 추가한다
- 난수 시드 42

## 코드 규칙

- Python 3.11+. 패키지는 `requirements.txt`
- 설정값(기간, fold, 거리, 임계값, 경로)은 `src/forecast/config.py` 한 곳에 모은다
- 인코딩: 먼저 `utf-8-sig`, 실패하면 `cp949`. 법정동 경계(umd.shp)는 cp949
- 중간 산출물은 parquet, 사람용 결과는 `reports/`에 md + png
- 그림의 한글 폰트는 시스템에 있는 것(AppleGothic, Malgun Gothic, NanumGothic 등)을 찾아 설정한다
- 사용자에게 보이는 문구는 짧고 평이하게 쓴다. 한자어 남발과 장황한 설명을 피한다
- 큰 파일(> 50MB)은 커밋하지 않는다

## 사용자에게 먼저 물어볼 것

- `data/interim/`의 음식점 파일이 없을 때
- 개폐업 동 수준에서 일반·휴게음식점을 합칠지 나눌지 바꾸고 싶을 때 (기본: 합침)
- 프랜차이즈 판별을 브랜드 목록으로 할지, 동일 상호 빈도 규칙으로 할지
- `forecast_plan.md`의 기본값(개폐업 많은 동 기준 월평균 10건, Croston 적용 0 비율 30%)을 바꿔야 할 때
