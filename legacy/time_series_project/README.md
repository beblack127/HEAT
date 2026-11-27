# HEAT: Seoul Hotspot Emergence Time-Series Pilot

## Current mobility forecasting experiment (2026-11-16)

The current presentation experiment forecasts **weekly average daily destination mobility** for ages 20–39 and purposes 4+5+7, by administrative dong, for the next **13 weeks**. Mobility is the only forecast target; Naver search is a lagged predictor. At the fixed 2025-04-27 origin, all methods were compared on the same 419 dongs and 13 weeks. Pooled WAPE was 8.52% for the prior-year same-week baseline, 18.93% for SARIMAX with lagged mobility, and 16.28% after adding Naver search. The search input improved this SARIMAX comparison but did not beat the simple annual baseline.

Start with [`outputs/nameplate_mobility_pilot/fixed_origin_search/README.md`](outputs/nameplate_mobility_pilot/fixed_origin_search/README.md) for the experiment and [`outputs/nameplate_mobility_pilot/ARIMA_FAMILY_PRESENTATION.md`](outputs/nameplate_mobility_pilot/ARIMA_FAMILY_PRESENTATION.md) for interpretation. Figures, case diagnostics, scripts, and presentation CSVs are in [`outputs/nameplate_mobility_pilot/`](outputs/nameplate_mobility_pilot/). Three large prediction CSVs are stored as `.csv.gz` without changing their contents; per-dong fit caches and raw API response batches are omitted. The older place-score pilot documented below is retained as prior work and does not describe the current forecast target.

## Earlier place-score pilot (archived context)

아래는 이전 장소 단위 점수 파일럿의 **실험 기록**입니다. 2023년 1월부터 2025년 12월까지의 자료를 사용했고, 2025년 네 개 분기에서 순차 백테스트했습니다. 이 저장소에는 수집·공간 연결·예측·점수 산출 코드, 수집 자료와 요약본, 모델 결과, 발표용 그림이 포함되어 있습니다.

## 먼저 읽을 자료

- [`results/semester_pilot/PROJECT_DECISION_2026-11-08.md`](results/semester_pilot/PROJECT_DECISION_2026-11-08.md): 당시 권고 파이프라인, 음식점 변수 비교, 홍대 계산 예시.
- [`results/semester_pilot/FIVE_STREAM_RESULTS.txt`](results/semester_pilot/FIVE_STREAM_RESULTS.txt): 검색량·상권 유동인구·생활인구·생활이동·지하철의 개별 예측과 결합 결과.
- [`results/semester_pilot/RESULTS.txt`](results/semester_pilot/RESULTS.txt): 최초 검색량·상권 유동인구 파일럿 기록. 일부 설명은 후속 보고서보다 이전 상태입니다.
- [`results/semester_pilot/five_stream_pilot.sqlite`](results/semester_pilot/five_stream_pilot.sqlite): 결합 패널과 연결표, 예측 결과를 담은 SQLite 파일.

## 발표 계획과 현재 파일럿의 차이

원래 발표자료의 Dataset 1~4는 **목적별 생활이동, 생활인구, 음식점 인허가, 지하철**입니다. 발표자료에서 검토 중이던 네이버 검색량은 실제로 수집했습니다. 당시 파일럿은 이와 별도로 **상권 추정 유동인구**를 주 방문활동 지표로, **상권 추정매출**을 외부 검증용 대리 지표로 썼습니다. 음식점 개·폐업은 원래 인허가 원자료를 지역별로 집계한 것이 아니라, **상권분석서비스 점포 자료**의 음식 업종을 사용한 예비 실험입니다. 인허가 원자료와 동일하다고 해석하면 안 됩니다.

| 분석 자료 | 공식 출처 | 파일럿에서의 역할 |
|---|---|---|
| 목적별 생활이동 | [OA-22298](https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do) | 행정동 주변 맥락; 별도 예측 |
| 생활인구 | [OA-14991](https://data.seoul.go.kr/dataList/OA-14991/F/1/datasetView.do) | 행정동 주변 맥락; 별도 예측 |
| 음식점 인허가(원안) | [일반음식점 OA-23056](https://data.seoul.go.kr/dataList/OA-23056/S/1/datasetView.do), [휴게음식점 OA-16095](https://data.seoul.go.kr/dataList/OA-16095/S/1/datasetView.do) | 지역별 원자료 결합은 **아직 하지 않음** |
| 지하철 역별 승하차 | [OA-12914](https://data.seoul.go.kr/dataList/OA-12914/F/1/datasetView.do) | 역 주변 맥락; 별도 예측 |
| 네이버 검색어 트렌드 | [NAVER API HUB](https://api.ncloud-docs.com/docs/naver-api-hub-search-trend) | 주 점수·예측 지표; 상대 지수 |
| 상권 추정 유동인구 | [OA-15568](https://data.seoul.go.kr/dataList/OA-15568/S/1/datasetView.do) | 주 점수·예측 지표; 실제 방문자 계수가 아닌 추정치 |
| 상권분석서비스 점포 | [OA-15577](https://data.seoul.go.kr/dataList/OA-15577/S/1/datasetView.do) | 음식업 점포·개업·폐업 선행 변수 실험 |
| 상권 추정매출 | [OA-15572](https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do) | 점수의 외부 대리 검증 |

공간 연결에는 [서울시 주요 121장소 경계](https://data.seoul.go.kr/dataList/OA-21285/F/1/datasetView.do), [상권 경계](https://data.seoul.go.kr/dataList/OA-15560/S/1/datasetView.do), [지하철역 좌표](https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=36)를 사용했습니다. 출처와 갱신 시점은 각 수집 코드 및 매니페스트를 참고하세요. 서울시 원천 자료를 인용할 때는 서울 열린데이터광장 및 해당 제공기관을 표시하세요.

## 이전 장소 단위 점수 파일럿의 파이프라인

```text
지역명·검색어 사전 ────── 네이버 월별 상대 검색지수 ──┐
공식 장소 경계 ─┬─ 상권 경계와 교차 ─ 상권 추정 유동인구 ─┼─ 개별 예측 ──┐
               │                    점포/개업/폐업 ──────┘             │
               ├─ 행정동과 교차 ──── 생활인구·생활이동 ─ 별도 예측 ─────┤
               └─ 주변 500m 역 ──── 지하철 하차 ───── 별도 예측 ───────┤
                                                                     ↓
                                  검색량+상권 유동인구 기본 점수 및 순위
                                  다섯 지표 결합은 비교 실험
                                                                     ↓
                                  상권 음식업 매출건수로 외부 대리 검증
```

기본 점수는 `0.35×검색 수준 백분위 + 0.15×검색 전년 증가율 백분위 + 0.35×상권당 유동인구 수준 백분위 + 0.15×상권당 유동인구 전년 증가율 백분위`입니다. 백분위 기준은 2024년 관측 분포로 고정했습니다. 80점, 두 분기 지속 조건을 시험했으나 2025년 신규 발생 사례는 없었습니다. 현재 결과는 신흥 핫플 **확정 판정**보다 다음 분기 **후보 순위 예측**으로 해석해야 합니다.

검색량은 31개 지역명, 주변 맥락 지표는 19개 장소에 연결됐습니다. 상권 경계와 결측 기준까지 통과하여 기본 점수를 비교한 장소는 10곳입니다. 100개 행정동 임시 목록은 장소명과 동일한 모집단위가 아니므로 주 점수 패널에 섞지 않았습니다. `data/admin_dong_sample/`의 설명을 읽고 사용하세요.

## 실행

Python 환경에서 `pip install -r requirements.txt` 후 프로젝트 루트에서 실행합니다. 데이터와 중간 결과는 저장소에 포함돼 있어 API 키 없이 기존 결과를 열어볼 수 있습니다. 새 네이버 검색 수집을 실행할 때만 본인의 NAVER API HUB 키가 필요하며, 수집기는 실행 중 입력을 받고 파일에 저장하지 않습니다.

1. 공간 연결 및 기본 패널: `build_place_dong_crosswalk.py` → `build_semester_panel.py`
2. 기본 예측·점수: `evaluate_semester_models.py` → `score_semester_pilot.py` → `evaluate_hotness_forecast.py` → `audit_hotness_score.py`
3. 주변 맥락: `build_subway_place_context.py` → `build_place_context_monthly.py` → `evaluate_context_streams.py` → `evaluate_late_fusion.py` → `audit_fusion_ablation.py`
4. 음식점 변수 분리 비교: `evaluate_restaurant_signal.py`
5. SQLite 재구성: `assemble_five_stream_database.py`

원천 재수집 코드는 `collect_*.py`, 네이버 시험용 코드는 `naver_*.py`입니다. 일부 코드는 서울시 공개 파일 형식 및 웹 다운로드 방식에 의존해 향후 사이트 변경 시 수정이 필요합니다.

## 핵심 파일럿 결과와 한계

- 검색량의 전년 동월 반복 3개월 예측 WAPE: **19.32%**.
- 상권 추정 유동인구의 마지막 공개 분기 반복 예측 WAPE: **5.31%**. 음식점 점포 변수를 넣은 Ridge는 **5.90%**로 개선되지 않았습니다.
- 10개 장소 × 2025년 4분기에서 기본 점수 예측 순위상관 평균 **0.901**, Precision@3 **0.750**.
- 다섯 지표 결합 점수는 현재 외부 매출 대리 지표와의 관련성을 높이지 못했습니다. 상세 수치는 보고서에 있습니다.

장소 경계의 2026년 배포본을 2023~2025년 자료에 잠정 적용했습니다. 행정동 전체와 역 주변의 수치는 해당 장소 방문자 수가 아닙니다. 네이버 검색지수도 절대 검색 건수가 아닙니다. TFT는 현재 점수 대상 패널이 10개 장소 × 12분기로 작아 훈련하지 않았습니다.
