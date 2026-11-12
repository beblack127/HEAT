# HEAT
시계열적 요소를 종합하여 서울 시내의 핫플레이스를 정의하고 떠오르는 지역을 예측하는 시스템

한양대학교 시계열데이터분석및실습 Team #14. 이 저장소에는 F&B·검색 파트 작업이 들어 있다.

## 지금까지 한 것

| 작업 | 결과 | 설명 |
|---|---|---|
| 행정동 이름표 | `data/processed/nametags/`, `maps/행정동_이름표_확인지도.html` | 427개 행정동마다 사람들이 검색할 이름(동네·역·상권)과 가중치 |
| 네이버 검색량 | `data/processed/search/`, `maps/행정동_이름표_검색량_지도.html` | 이름표 566묶음 × 방문 의도어 6개, 2022-01~2026-10 월별, 동별 서울 대비 비중 |
| 음식점 전처리·EDA | `reports/preprocess/`, `reports/eda/`, `data/processed/fnb/` | 인허가 정제, 서울 전체 월별 개폐업 집계 |
| 예측 계획 (v3) | `docs/forecast_plan.md` | 2022~2025 학습 → 2026 예측, 3개월 기본 |

지도는 HTML 파일 하나로 열린다 (인터넷·API 키 필요 없음).

## 다음 할 것

`docs/forecast_plan.md` §8 순서대로 검색량·개폐업 예측 파이프라인(`src/forecast/`)을 만든다. 작업 규칙은 `CLAUDE.md`.

## 폴더

```
CLAUDE.md               Claude Code 작업 규칙
docs/                   계획(forecast_plan v3, analysis_plan v2), 지표 명세, 연구 근거
config/                 업종 매핑표
data/external/geo/      행정동(emd_20251231)·법정동(umd)·한강·상권 영역(sk)·역·랜드마크
data/raw/search_api/    네이버 API HUB 응답 원본
data/processed/         이름표, 검색량, 개폐업 월 집계
maps/                   단독 HTML 지도
src/nametags/           이름표·지도
src/search/             검색량 수집·동별 지수
src/preprocess/, src/eda/  인허가 전처리, 초기 EDA (원래 플젝 폴더 구조 기준 경로)
reports/                결과 요약 문서와 그림
```

## 다시 만들기

```bash
pip install -r requirements.txt

# 이름표 (data/interim/음식점_정리본.csv 필요)
bash src/nametags/run.sh

# 검색량 동별 지수와 지도 (저장된 API 응답으로 다시 계산, 키 필요 없음)
bash src/search/run.sh
```

결과는 `build/`에 생기며, 저장소에 있는 파일과 같게 나온다.

## 로컬에만 두는 파일

git에 올리지 않는다. 팀 폴더에서 받아 같은 위치에 둔다.

- `data/interim/음식점_정리본.csv`, `data/interim/음식점_개폐업사건.csv` — 전처리 결과 (각 100MB 이상)
- `data/raw/` — 인허가·상권분석서비스 원본
- `.env` — 네이버 API 키 (`NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`). 검색량을 새로 받을 때만 필요

## 출처

- 인허가: 서울 열린데이터광장 일반음식점(OA-23056)·휴게음식점(OA-16095)
- 행정동 경계: vuski/admdongkor (2025-12-31), 법정동 경계: V-World LSMD_ADM_SECT_UMD
- 상권 영역: 서울시 상권분석서비스(영역-상권), 역: 서울시 역사마스터 정보
- 한강: OpenStreetMap (ODbL)
- 검색량: NAVER API HUB 검색어 트렌드
