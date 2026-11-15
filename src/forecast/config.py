"""예측 파이프라인 설정. 기간·fold·거리·임계값·경로는 모두 여기서만 정한다 (docs/forecast_plan.md v3)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------- 경로 ----------------
SEARCH_CSV = ROOT / "data/processed/search/행정동_검색지수_월별.csv"
DONG_LIST_CSV = ROOT / "data/processed/nametags/01_행정동_목록.csv"
EMD_PARQUET = ROOT / "data/external/geo/emd_20251231.parquet"
STORES_CSV = ROOT / "data/interim/음식점_정리본.csv"
EVENTS_CSV = ROOT / "data/interim/음식점_개폐업사건.csv"
CATEGORY_CSV = ROOT / "config/category_mapping.csv"

INTERIM = ROOT / "data/interim/forecast"
RESULTS = ROOT / "results/forecast"
REPORTS = ROOT / "reports/forecast"
EXP_LOG = ROOT / "experiments/log.csv"
UNMAPPED_CSV = ROOT / "reports/unmapped_categories.csv"

# ---------------- 기간 ----------------
TRAIN_START = "2022-01"          # 학습 표본·국소 모델 적합 시작
HIST_START = "2015-01"           # F&B 피처 계산용 과거 (t 시점에 알 수 있던 값)
DEV_END = "2025-12"              # 개발(F2024·F2025) 단계에서 읽는 마지막 달. 2026 값은 최종 시험 전까지 보지 않는다
FINAL_END = "2026-09"            # 최종 시험 마지막 달 (2026-10은 부분 월이라 제외)
COVID_MONTHS = ("2022-01", "2022-04")   # covid = 1 구간 (양끝 포함)

HORIZONS = (3, 6, 12)
MAIN_H = 3
MAX_H = max(HORIZONS)

# fold: 학습 끝, 시험 연도 (origin은 학습 끝 ~ 시험 연도 12월 - h)
FOLDS = {
    "F2024": {"train_end": "2023-12", "test_year": 2024},
    "F2025": {"train_end": "2024-12", "test_year": 2025},
}
FINAL_FOLD = {"FINAL": {"train_end": "2025-12", "test_end": "2026-09"}}

# ---------------- 개폐업 ----------------
BULK_MULT = 10        # 같은 동·같은 폐업일 건수 >= 동 일평균 x 10
BULK_MIN = 10         #   그리고 >= 10건
BUSY_DONG_MIN = 10    # SARIMA 적용: 학습 구간 월평균 개업+폐업 >= 10
CROSTON_ZERO_SHARE = 0.30   # Croston/TSB 적용: 개업 또는 폐업 0인 달 비율 >= 30%
PANELS = ("v_drop", "v_keep")
MAIN_PANEL = "v_drop"

# ---------------- 평가 ----------------
TOPK = (10, 20, 30)
TOPK_SEARCH_MIN_LEVEL_PCT = 0.25   # 검색 변화 순위: origin 시점 수준 하위 25% 동 제외 (지도 규칙과 같음)
TOPK_NETRATE_MIN_STOCK = 30        # 순증률 순위: 영업 점포 30개 미만 동 제외 (작은 분모 잡음)

# ---------------- 모델 ----------------
SEED = 42
SEASON = 12
N_JOBS = -1
SEARCH_SCALE = 1e4    # 국소 모델 적합 시 비중 x 1e4 (수치 안정)
LGB_PARAMS = dict(num_leaves=15, min_data_in_leaf=50, learning_rate=0.03, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=2000,
                  seed=SEED, verbose=-1, num_threads=8)
EARLY_STOP_MONTHS = 3     # 학습 구간 마지막 3개월(실제값 끝 기준)을 early stopping 검증으로
EARLY_STOP_ROUNDS = 100
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
