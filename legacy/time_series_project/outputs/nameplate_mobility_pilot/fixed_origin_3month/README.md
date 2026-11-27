# 고정된 예측 시점에서 향후 3개월 예측

> **검색량 후속 실험:** [419개 동의 네이버 검색지수 추가 비교](../fixed_origin_search/README.md). 같은 고정 원점에서 수업 모델의 WAPE는 일부 낮아졌지만 전년 같은 주 기준선을 넘지 못했다.

## 바로잡은 예측 질문

**한 시점까지 수집된 생활이동만 사용해, 그 다음 3개월의 동별 이동량을 한 번에 예측한다.** 예측 도중 새로 관측된 실제값으로 모델 상태를 갱신하지 않는다. 앞서 만든 `mobility_only_dong/`의 2025년 예측은 각 주마다 새 실제값을 반영한 **rolling 13주 선행 예측**이므로 이 질문에 대한 결과가 아니다. 기존 수치와 그래프를 고정 원점 3개월 예측의 성능으로 인용하지 않는다.

- **고정 백테스트:** 2025-04-27까지 학습 → 2025-04-28~07-27의 **연속 13주**를 한 번에 예측. 실제값과 비교.
- **마지막 수집 시점:** 2025-08-31까지 학습 → 2025-09-01~11-30의 **연속 13주**를 한 번에 예측. 수집본에는 이 구간의 실제값이 없어 성능을 매길 수 없다. 현재 날짜 기준의 미래 예측도 아니다.
- 매 동마다 ARIMA, SARIMA, ARIMAX, SARIMAX를 따로 적합했다. ARIMAX/SARIMAX 외생변수는 **같은 동의 13주 전 생활이동**뿐이다. 다음 13주의 이 변수값은 예측 시점에 모두 이미 관측되어 있다. 네 모델은 같은 시작·종료 학습 구간을 사용한다.
- 도착 행정동 목적 4+5+7, 남녀 20·30대, 주간 일평균 **추정 이동량**, 완전한 139주가 있는 424개 동. 예측 대상은 고유 방문자 수가 아니다.

## 한 번에 13주를 예측한 동별 백테스트

WAPE = 424개 동 × 13개 미래 주의 절대오차 합 / 실제 이동량 합. 13주차 한 점도 별도 계산했다.

| 방법 | 다음 13주 전체 WAPE | 13주차 WAPE |
|---|---:|---:|
| **전년 같은 주 기준선** | **8.51%** | **8.29%** |
| ARIMA | 19.14% | 23.47% |
| SARIMA | 19.43% | 22.98% |
| ARIMAX | 19.31% | 22.50% |
| SARIMAX | 18.91% | 21.84% |

이 한 번의 백테스트에서는 네 수업 모델이 모두 단순 계절 기준선보다 훨씬 나쁘다. 네 모델이 특정 동의 최근 상승·하락을 앞으로 지속시켜 과대·과소 예측하는 사례가 있다. **백테스트 원점 하나만으로 일반 성능을 확정할 수는 없다.** 다른 시점의 고정 원점 검증, 2024년 8~10월 원자료 급증의 처리 근거가 추가로 필요하다. 예측 그래프에서 파란 실선은 원점까지 관측한 값, 빨간 선은 그 원점에서 한 번 만든 13주 예측, 파란 점선은 이후 확인된 실제값이다.

## 결과 파일

- `backtest_fixed_origin_example.png`: 청운효자동의 고정 원점 백테스트 그림. 동 코드순 첫 번째를 **사전에 정해** 표시했다.
- `final_fixed_origin_example.png`: 같은 동의 마지막 수집 시점으로 만든 2025년 9~11월 예측. 미래 실제값은 표시하지 않는다.
- `backtest_metrics.csv`, `backtest_predictions.csv`: 424동 평가 수치와 주별 예측.
- `final_predictions.csv`: 424동 × 13주 × 네 모델 예측값. 전년 같은 주 기준선도 포함.
- 재현 코드: `../run_fixed_origin_3month.py`.

## 발표 문장

> “Using data available at one cutoff, we forecast the next 13 weeks of destination mobility for each Seoul neighborhood. In an initial fixed-origin backtest, the prior-year weekly baseline outperformed all four class models. We are examining source anomalies and validation across additional cutoffs before claiming predictive gains.”
