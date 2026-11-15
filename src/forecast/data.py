"""예측 대상 계열을 넓은 표(행 = 월, 열 = 계열)로 꺼낸다.

대상 이름
  search            동별 검색 비중 (칸 '전체'). 예측값 = t+1..t+h 평균
  open              개업 수. 예측값 = t+1..t+h 합
  close_v_drop      폐업 수 (일괄 폐업 제외)
  close_v_keep      폐업 수 (일괄 폐업 포함)
파생 (06_evaluate에서 계산): net_rate_<panel> = (ΣO − ΣC) / S_t, churn_<panel> = (ΣO + ΣC) / S_t
"""
import pandas as pd

from common import C, load_monthly, mrange

COUNT_TARGETS = ("open", "close_v_drop", "close_v_keep")
AGG = {"search": "mean", "open": "sum", "close_v_drop": "sum", "close_v_keep": "sum"}
LEVELS = {"search": ("dong",), "open": ("seoul", "gu", "dong"),
          "close_v_drop": ("seoul", "gu", "dong"), "close_v_keep": ("seoul", "gu", "dong")}
COL = {"open": "O", "close_v_drop": "C_v_drop", "close_v_keep": "C_v_keep"}

_cache = {}


def monthly(final=False):
    if final not in _cache:
        _cache[final] = load_monthly(final)
    return _cache[final]


def wide(target, level="dong", final=False, start=None):
    s, f = monthly(final)
    if target == "search":
        x = s[s.tier == "전체"].pivot(index="month", columns="dong_code8", values="share")
    else:
        g = f[f.level == level]
        x = g.pivot(index="month", columns="series_id", values=COL[target])
    if start:
        x = x.loc[start:]
    return x.sort_index()


def stock(level="dong", final=False):
    _, f = monthly(final)
    return f[f.level == level].pivot(index="month", columns="series_id", values="S").sort_index()


def fnb_wide(col, level="dong", final=False):
    _, f = monthly(final)
    return f[f.level == level].pivot(index="month", columns="series_id", values=col).sort_index()


def search_wide(col="share", tier="전체", final=False):
    s, _ = monthly(final)
    return s[s.tier == tier].pivot(index="month", columns="dong_code8", values=col).sort_index()


def gu_of(final=False):
    _, f = monthly(final)
    d = f[f.level == "dong"].drop_duplicates("series_id")
    return d.set_index("series_id").gu


def h_agg(x, h, how):
    """t+1..t+h 합 또는 평균 (행 t에 놓음)."""
    r = x[::-1].rolling(h, min_periods=h)
    r = (r.sum() if how == "sum" else r.mean())[::-1]
    return r.shift(-1)


def months_all(final=False):
    return mrange(C.TRAIN_START, C.FINAL_END if final else C.DEV_END)
