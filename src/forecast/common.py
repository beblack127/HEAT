"""공용 함수: 월 계산, fold origin, 대상 테이블 읽기, 실험 기록."""
import csv
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C  # noqa: E402


def P(m):
    return pd.Period(m, freq="M")


def mrange(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def shift_m(m, k):
    return str(P(m) + k)


def origins(fold, h):
    """fold의 평가 origin: 학습 끝 ~ (시험 연도 12월 - h). 최종 시험은 2026-09 - h까지."""
    if fold == "FINAL":
        f = C.FINAL_FOLD["FINAL"]
        return mrange(f["train_end"], shift_m(f["test_end"], -h))
    f = C.FOLDS[fold]
    return mrange(f["train_end"], shift_m(f"{f['test_year']}-12", -h))


def train_end(fold):
    return C.FINAL_FOLD["FINAL"]["train_end"] if fold == "FINAL" else C.FOLDS[fold]["train_end"]


def data_end(final=False):
    return C.FINAL_END if final else C.DEV_END


def covid(months):
    m = pd.Series(months).astype(str)
    return ((m >= C.COVID_MONTHS[0]) & (m <= C.COVID_MONTHS[1])).astype(int).values


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", **kw)


def load_monthly(final=False):
    """모델링용 월 패널. 개발 단계에서는 DEV_END(2025-12)까지만 돌려준다."""
    end = data_end(final)
    s = pd.read_parquet(C.INTERIM / "search_monthly.parquet")
    f = pd.read_parquet(C.INTERIM / "fnb_monthly.parquet")
    return s[s.month <= end].copy(), f[f.month <= end].copy()


def future_sum(wide, h):
    """wide(행=월, 열=계열)에서 t+1..t+h 합 (행 t에 놓음)."""
    return wide[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


def past_sum(wide, h, lag=0):
    """t-h+1-lag .. t-lag 합."""
    return wide.rolling(h, min_periods=h).sum().shift(lag)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def log_run(step, note, **kw):
    """experiments/log.csv에 한 줄 추가."""
    C.EXP_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not C.EXP_LOG.exists()
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "step": step, "git": git_hash(),
           "spec_version": "2.2", "plan": "v3", "seed": C.SEED, "note": note,
           "extra": "; ".join(f"{k}={v}" for k, v in kw.items())}
    with open(C.EXP_LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.name}] 시작", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[{self.name}] {time.time() - self.t:.1f}초", flush=True)


def set_korean_font():
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans Gothic"):
        if n in names:
            matplotlib.rcParams["font.family"] = n
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out
