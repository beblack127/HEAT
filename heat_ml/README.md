# HEAT 2026 주간 생활이동 예측

작업공간 루트에서 PowerShell로 실행한다. 입력은 읽기 전용이며 생성물은 `heat_ml/` 아래에 저장된다.

```powershell
python -m pip install -r heat_ml/requirements.txt
python -m heat_ml smoke
python -m heat_ml smoke --full --out heat_ml/smoke_run
python -m heat_ml prepare
python -m heat_ml run --releases heat_ml/releases.csv --common heat_ml/common.csv --subway heat_ml/subway_daily.parquet --out heat_ml/run_2026 --jobs 4
```

`smoke`는 합성 자료로 날짜 경계, 학습 라벨, 보정, 공개일, WAPE 계산을 확인한다. `smoke --full`은 축소된 Ridge·LightGBM 그리드와 계절 기준값을 사용해 출력까지 검사한다. 두 명령 모두 실제 성능을 산출하지 않는다.

## 입력

- **생활이동** — `run --target`은 CSV 또는 Parquet의 `unit,week_end,y,n_days,seoul_y`를 읽는다. 기본 순서는 `HEAT/data/interim/gt/target_weekly.parquet`, 그다음 `heat_ml/target_2023_2025.csv`다. `prepare`는 `data/seoul_purpose_admdong1_in_YYYYMM.zip` 내 일별 CSV를 읽어 목적 4·5·7의 남녀 20대·30대 합, HEAT 동 코드 통합, 일요일 종료 주의 일평균을 계산한다. 관측일 4일 미만인 동·주는 결측이다. 서울 합계는 동 매핑 전 일별 합으로 계산한다. 기존 주간 합계 CSV는 통합하기 전의 일별 관측 집합을 복원할 수 없으므로 원본 ZIP을 사용한다.
- **공통 동** — `--common`은 정확한 HEAT 동 집합을 담은 `unit` 열을 받는다. 시점별 집합인 경우 `origin` 열을 추가한다. 기본 파일은 `HEAT/results/gt/unit_metrics.parquet`다. 네 시점 각각 416개의 알려진 동을 검증하며, 모든 동으로 적합한 뒤 공통 동에서 평가한다. `--historical-rows`에 t7의 실제 `origin,unit,week_end` 행 목록을 제공하면 그 목록으로 비교를 제한하고 예측이나 정답이 빠진 행을 보고한다.
- **검색 및 F&B** — 기본값은 HEAT의 `행정동_검색지수_월별.csv`와 `행정동_개폐업_월별.csv`다. 검색은 `dong_code8,tier,month,seoul_share`, F&B는 `level,code,month,개업,폐업_일괄제외,영업점포`를 사용한다. 목표 주 종료일의 4개월 전 달을 조회한다.
- **지하철** — 기본값은 `HEAT/data/raw/subway/subway_daily.parquet`다. `--subway`에 `station,date,alight` 열이 있는 CSV 또는 Parquet을 전달할 수 있다. HEAT 역 연결표와 면적 가중치를 재사용해 13주 전 종료일의 4주 평균 하차량을 구한다.
- **공개일** — `--releases`는 `feature,period,available_on` 열을 받는다. `feature`는 `search`, `fnb`, `subway` 중 하나다. `period`는 월말 또는 일요일이며 날짜는 `YYYY-MM-DD` 형식이다. 실제 입력값의 최초 이용 가능일을 기록하고, 개정 자료의 경우 당시 제공된 값과 그 값의 공개 기록을 함께 사용한다. 최근에 수집된 파일의 과거 값에 당시 공개일을 추정해 부여하면 실시간 비교를 재현할 수 없다. 공개일이 없거나 origin보다 늦은 값과 롤링 계산의 구성 요소는 결측 처리한다.

공통 동 또는 외생 자료가 빠져도 명시적으로 축소 실험을 실행할 수 있다. 이 경우 실제 공통 416동 실험과 구분해 기록한다.

```powershell
python -m heat_ml run --conditions base --allow-partial --out heat_ml/run_base --jobs 4
```

## 검증 및 산출물

### 1차 기본 피처 실험

```powershell
python -m heat_ml run --phase1 --allow-partial --out heat_ml/run_phase1 --jobs 6
```

행정동과 예측 거리만 쓰는 최소 조건, 시차값·이동평균·전년 동주·푸리에 항을 각각 추가한 네 조건, 전체 기본 피처 조건, 전체에서 각 묶음을 하나씩 뺀 네 조건을 비교한다. 정확한 기존 동 목록이 있으면 `--common`으로 지정한다. 목록이 없으면 관측값과 예측값이 모두 있는 공통 행에서 평가하며 결과에 `phase1_partial`을 기록한다.

LightGBM은 `(num_leaves=15, min_data_in_leaf=200, learning_rate=0.03)`에서 시작해 잎 수, 잎의 최소 표본 수, 학습률 순으로 비교한다. 같은 조합의 내부 검증 결과를 재사용한다. 단독 예측의 4시점 평균 WAPE가 가장 낮은 LightGBM 조건 두 개에 전체 12개 후보를 다시 적용하고, `lightgbm_full_grid_check.csv`에 단계별 탐색과 비교한 결과를 기록한다. `horizon_wape.csv`에는 1~4주, 5~8주, 9~13주 WAPE를 기록한다. 2026년 예측 생성은 최종 모형 선택 단계에서 수행한다.

- 각 외부 cutoff에서 직전 두 개의 독립된 13주 검증 블록으로 확장창 튜닝을 한다. 원 단위 WAPE로 파라미터와 LightGBM 조기 종료를 선택하고, 선택된 설정의 두 반복 수의 중앙값으로 재학습한다. 자기 시차의 1주는 origin 종료일에서 7일 전이며 직전 이동평균은 origin 주를 포함한다. 역사적 학습 특성은 각 행의 origin에서 관측할 수 있었던 보정계수를 사용하며, 학습 정답은 분할 cutoff의 보정계수를 사용한다.
- `outer_predictions.csv`, `coverage.csv`, `dong_wape.csv`에 행별 예측, 누락과 제외, 동별 WAPE와 기준선 차이를 기록한다. `outer_wape.csv`와 `mean_wape.csv`는 모든 후보와 재적합 기준선이 공유하는 유효 행에서 시점별 원자료 WAPE, 4시점 평균, 2시점 평균을 기록한다. t7의 회고적 수치는 `selection.json`에 별도 참고치로 기록한다. 원본 t7 표의 보정 ARIMA 결합 2시점 평균은 10.69%다.
- 4시점 평균이 가장 낮은 ML 후보를 선택한다. 동률이면 2시점 평균을 사용한다. 2025-12-28까지 다시 튜닝·적합하여 그다음 13주를 `final_forecast.csv`에 고정한다. `selection.json`과 `outer_tuning.json`에 파라미터, 내부 검증 점수, 보정계수를 기록한다.
- 2026년 실제값이 들어오면 같은 형식의 주별 타깃을 `--truth`로 전달한다. 새로 실행할 때 `run --truth`를 지정해도 예측 파일이 저장된 다음에만 채점한다. 채점 결과에는 유효 행과 주·동별 자료 범위가 포함된다.

```powershell
python -m heat_ml prepare --mobility-dir data --first 2025-12-29 --last 2026-03-29 --output heat_ml/truth_2026.csv
python -m heat_ml score --forecast heat_ml/run_2026/final_forecast.csv --truth heat_ml/truth_2026.csv --out heat_ml/run_2026
```

원본 ZIP에서 이 기간을 만들려면 2025년 12월과 2026년 1~3월 파일이 필요하다.
