# src/gt_refine — 정답(GT) 다듬기 실험 (재민)

중간발표 이후 정답 정의를 점검한 실험. 결과는 `reports/gt_refine/`에 있다.
원래 `JAEMINII/Time-series-Project`의 `jaemin` 브랜치에서 만들었고(커밋 896604c, 5c64f72), 2026-11-27에 이 저장소로 옮겼다.

| 스크립트 | 내용 | 결과 |
|---|---|---|
| `run_gt_share_change.py` | 서울 대비 비중의 1년 전 대비 변화(g)를 정답으로 쓸 수 있는지 점검: 모멘텀 기준선, 잡음, 목록 안정성, 2024-08~10 이상치, 20~39세 vs 전 연령, 카드 매출·검색 변화와의 일치 | `reports/gt_refine/share_change/` |
| `run_card_hotplace.py` | 20~39세의 핫플 업종 카드 매출로 동 순위를 만들고 서울시 주요 121장소(발달상권·관광특구)와 비교 | `reports/gt_refine/card_hotplace/` |

## 입력 (git에 없음, 로컬에만 있음)

두 스크립트는 첫 인자 `<codex_root>` 아래의 경로를 읽는다. 이 저장소의 `data/` 구조로는 아직 옮기지 않았다.

- `<codex_root>/outputs/nameplate_mobility_pilot/age2030_daily_baseline/daily_457_age2030_YYYYMM.csv`: 생활이동 20~39세 4+5+7 일별(동 × 일), 2023-01~2025-08
- `<codex_root>/outputs/nameplate_mobility_pilot/age2030_mobility/age2030_457_YYYYMM.csv`: 같은 자료의 월 집계
- `<codex_root>/work/Time-series-Project/data/reference/`
  - `seoul_commercial_sales_{2023,2024,2025}.zip`: 상권분석서비스 추정매출(상권, OA-15572)
  - `seoul_commercial_area_2023.zip`: 상권 영역(상권 → 행정동)
  - `seoul_121_place_boundaries/`: 서울시 주요 121장소 경계
- 두 번째 인자 `<heat_root>`: 이 저장소. `data/processed/search/행정동_검색지수_월별.csv`, `data/processed/nametags/01_행정동_목록.csv`, `_mapdata_small.json`을 읽는다

```bash
python src/gt_refine/run_gt_share_change.py <codex_root> . reports/gt_refine/share_change
python src/gt_refine/run_card_hotplace.py <codex_root> . reports/gt_refine/card_hotplace
```

## 주의

- 두 스크립트는 pandas를 쓴다. 2026-11-25부터 재민 PC에서는 pandas가 보안 정책(애플리케이션 제어)에 막혀, 옮긴 뒤 다시 돌려 보지는 못했다. 코드는 원래 저장소에서 결과를 만든 그대로다.
- 카드 매출은 2023~2025년만 쓴다(2021~2022년은 코로나 기간이라 제외).
