"""생활이동 정답 모델 설정 (docs/experiments/midterm_experiments.md §2)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

# 경로
RAW_MOB = ROOT / "data/raw/mobility"        # 줄인 생활이동 (동 x 일 x 목적 x 연령대)
RAW_SUB = ROOT / "data/raw/subway"          # 줄인 지하철 (역 x 일 하차)
RAW_LP = ROOT / "data/raw/living_pop"       # 줄인 생활인구 (동 x 일 주간 평균)
INTERIM = ROOT / "data/interim/gt"
RESULTS = ROOT / "results/gt"
REPORT = ROOT / "reports/gt_midterm"
FIG = REPORT / "fig"
TABLES = REPORT / "tables"
DONG_LIST = ROOT / "data/processed/nametags/01_행정동_목록.csv"
STA_LINK = ROOT / "data/processed/nametags/05_행정동_역_연결표.csv"
SEARCH_CSV = ROOT / "data/processed/search/행정동_검색지수_월별.csv"
FNB_CSV = ROOT / "data/processed/fnb/행정동_개폐업_월별.csv"
EMD = ROOT / "data/external/geo/emd_20251231.parquet"
EXP_LOG = ROOT / "experiments/log.csv"

# 정답
TARGET_AGES = ("20", "30")
TARGET_PURPOSES = (4, 5, 7)
SENS_PURPOSES = (4, 5)
DATA_START = "2023-01-02"     # 첫 월요일
MIN_DAYS = 4                  # 주 안 유효 일수가 이보다 적으면 결측

# 예측
ORIGINS = ["2024-10-27", "2025-01-26", "2025-04-27", "2025-07-27"]   # 일요일
H = 13
SEASON = 52
X_SETS = {
    "none": [],
    "search": ["search_share"],
    "all_noLP": ["search_share", "fnb_stock_chg12", "subway_exit_4w", "holiday_days"],
    "all": ["search_share", "fnb_stock_chg12", "subway_exit_4w", "holiday_days", "living_pop"],
    "subway": ["subway_exit_4w"],                       # 추가 실험 (2026-11-16): 지하철만
    "search_subway": ["search_share", "subway_exit_4w"],  # 추가 실험: 검색 + 지하철 (생활인구 제외)
    "fnb": ["fnb_net3"],                                  # 추가 실험: 음식점 순개업률 (3개월, 4개월 전 달)
    "search_fnb": ["search_share", "fnb_net3"],
    "subway_fnb": ["subway_exit_4w", "fnb_net3"],
    "search_subway_fnb": ["search_share", "subway_exit_4w", "fnb_net3"],
}
PRIOR_YEAR_X = "y_lag52"
LAG_WEEKLY_X = 13
LAG_MONTHLY_X = 4

# 핫스팟
HOT_TOP = 0.10
EMERGING_MIN_SIZE_PCT = 0.25   # 규모 하한 = 기준 연도 동별 평균의 하위 25% 경계 (고정값)
EMERGING_MIN_BASE_YEAR = 2023  # 이 해의 동별 주 평균 분포로 하한을 한 번만 정한다
TOPK = (10, 20, 30)
CASE_DONGS = ["화곡1동", "이촌2동"]
DRAFT_419 = {"snaive": 8.52, "sarimax_pastmob": 18.93, "sarimax_pastmob_search": 16.28}   # 중간발표 초안 (2025-04-27)
SEED = 42
for p in (INTERIM, RESULTS, FIG, TABLES): p.mkdir(parents=True, exist_ok=True)
