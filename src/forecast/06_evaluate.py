"""평가: MASE(계절 naive 분모), Poisson deviance, top-K 적중률, DM 검정(HAC 시차 h−1, HLN 보정).

대상
  search                검색 비중 h개월 평균 (동)
  open, close_<panel>   개업·폐업 h개월 합 (서울·구·동)
  net_rate_<panel>      Σ(O−C)/S_t. 국소·기준선 모델은 개업·폐업 예측에서 계산, 전역 모델은 '·직접' 버전도 있음
  churn_<panel>         Σ(O+C)/S_t (보조)
비교 범위(scope): 동 수준 개폐업은 all(전체 동), busy(개폐업 많은 동, SARIMA·SARIMAX), zero(0 많은 동, Croston·TSB).
한 범위 안에서는 모든 모델이 같은 계열·origin을 갖도록 맞춘다.
출력: reports/forecast/06_metrics.csv, 06_topk.csv, 06_dm.csv, 06_results.md, 그림
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import C, log_run, mrange, origins, set_korean_font, shift_m, train_end
import data as D

FINAL = "--final" in sys.argv
FOLDS = ["FINAL"] if FINAL else list(C.FOLDS)
set_korean_font()
R = C.REPORTS
SUF = "_final" if FINAL else ""

# ---------------- 예측 모으기 ----------------
parts = []
for f in sorted(C.RESULTS.glob(f"pred_*{SUF}.parquet")):
    if not FINAL and f.stem.endswith("_final"):
        continue
    parts.append(pd.read_parquet(f))
pred = pd.concat(parts, ignore_index=True)[["target", "level", "series_id", "fold", "origin", "h", "model", "y_pred"]]
pred = pred[pred.fold.isin(FOLDS)]
pred["series_id"] = pred.series_id.astype(str)
# 전역 모델의 순증률 직접 예측은 이름을 구분
direct = pred.target.str.startswith("net_rate")
pred.loc[direct, "model"] = pred.loc[direct, "model"] + "·직접"

# 파생: 순증률·회전율 = 개업·폐업 예측 / S_t
key = ["level", "series_id", "fold", "origin", "h", "model"]
op = pred[pred.target == "open"].set_index(key).y_pred
deriv = []
for panel in C.PANELS:
    cl = pred[pred.target == f"close_{panel}"].set_index(key).y_pred
    j = pd.concat([op.rename("O"), cl.rename("C")], axis=1, join="inner").reset_index()
    for lv in j.level.unique():
        S = D.stock(lv, FINAL).stack().rename("S_t")
        S.index.names = ["origin", "series_id"]
        jj = j[j.level == lv].join(S, on=["origin", "series_id"])
        deriv.append(jj.assign(target=f"net_rate_{panel}", y_pred=(jj.O - jj.C) / jj.S_t.where(jj.S_t > 0)))
        deriv.append(jj.assign(target=f"churn_{panel}", y_pred=(jj.O + jj.C) / jj.S_t.where(jj.S_t > 0)))
pred = pd.concat([pred] + [d[pred.columns] for d in deriv], ignore_index=True)


# ---------------- 실제값 ----------------
def actual_wide(target, level, h):
    """행 = origin, 값 = t+1..t+h 실제값."""
    if target == "search" or target in D.COUNT_TARGETS:
        return D.h_agg(D.wide(target, level, FINAL), h, D.AGG[target])
    panel = target.split("_", 2)[-1] if target.startswith("net_rate") else target.split("_", 1)[1]
    O = D.h_agg(D.wide("open", level, FINAL), h, "sum")
    Cc = D.h_agg(D.wide(f"close_{panel}", level, FINAL), h, "sum")
    S = D.stock(level, FINAL).reindex(O.index)[O.columns]
    S = S.where(S > 0)
    return (O - Cc) / S if target.startswith("net_rate") else (O + Cc) / S


def snaive_wide(target, level, h):
    """계절 naive 예측 (MASE 분모용, 학습 구간)."""
    def sn(x, how):
        r = x.rolling(h, min_periods=h)
        return (r.sum() if how == "sum" else r.mean()).shift(12 - h)
    if target == "search" or target in D.COUNT_TARGETS:
        return sn(D.wide(target, level, FINAL), D.AGG[target])
    panel = target.split("_", 2)[-1] if target.startswith("net_rate") else target.split("_", 1)[1]
    O = sn(D.wide("open", level, FINAL), "sum")
    Cc = sn(D.wide(f"close_{panel}", level, FINAL), "sum")
    S = D.stock(level, FINAL).reindex(O.index)[O.columns]
    S = S.where(S > 0)
    return (O - Cc) / S if target.startswith("net_rate") else (O + Cc) / S


def denom(target, level, fold, h):
    """계열별 MASE 분모: 학습 구간 origin(1년 전 값이 TRAIN_START 이후, 실제값 끝 <= 학습 끝)에서 계절 naive MAE."""
    a, s = actual_wide(target, level, h), snaive_wide(target, level, h)
    tr = mrange(shift_m(C.TRAIN_START, 11), shift_m(train_end(fold), -h))
    return (a.loc[tr] - s.loc[tr]).abs().mean()


targets = sorted(pred.target.unique())
rows_ev = []
for target in targets:
    for level in pred[pred.target == target].level.unique():
        for h in C.HORIZONS:
            A = actual_wide(target, level, h)
            A.columns = A.columns.astype(str)
            a = A.stack().rename("y_true")
            a.index.names = ["origin", "series_id"]
            for fold in FOLDS:
                den = denom(target, level, fold, h)
                den.index = den.index.astype(str)
                p = pred[(pred.target == target) & (pred.level == level) & (pred.h == h) & (pred.fold == fold)]
                if p.empty:
                    continue
                p = p.join(a, on=["origin", "series_id"])
                p["den"] = p.series_id.map(den)
                rows_ev.append(p)
ev = pd.concat(rows_ev, ignore_index=True)
ev = ev[ev.y_true.notna() & ev.y_pred.notna()]
ev.to_parquet(C.RESULTS / f"eval_rows{SUF}.parquet", index=False)

# ---------------- 비교 범위 ----------------
scopes = {fold: json.load(open(C.INTERIM / f"scope_{fold}.json")) for fold in FOLDS}


def scope_rows(e, fold, scope):
    if scope == "all":
        return e
    ids = set(map(str, scopes[fold]["busy" if scope == "busy" else "croston"]))
    return e[e.series_id.isin(ids)]


def common_subset(e):
    """모든 모델이 같은 (계열, origin)을 갖도록 교집합."""
    n_models = e.model.nunique()
    k = e.groupby(["series_id", "origin"]).model.nunique()
    keep = k[k == n_models].index
    return e.set_index(["series_id", "origin"]).loc[keep].reset_index()


def pois_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def model_sets(e, level, target):
    """범위별로 비교할 모델 묶음. 범위에 다 들어있지 않은 모델은 빼고 공통 부분집합을 만든다."""
    out = []
    is_cnt_dong = level == "dong" and target != "search"
    for scope in (["all", "busy", "zero"] if is_cnt_dong else ["all"]):
        ms = set(e.model.unique())
        if scope == "all" and is_cnt_dong:
            ms -= {"SARIMA", "SARIMAX", "Croston", "TSB"}
        if scope == "busy":
            ms -= {"Croston", "TSB"}
        if scope == "zero":
            ms -= {"SARIMA", "SARIMAX"}
        out.append((scope, sorted(ms)))
    return out


metrics, topk, dms, loss_store = [], [], [], {}
for (target, level, fold, h), e0 in ev.groupby(["target", "level", "fold", "h"]):
    for scope, ms in model_sets(e0, level, target):
        e = scope_rows(e0[e0.model.isin(ms)], fold, scope)
        # SARIMAX는 h=3,6만: 해당 h에 없는 모델은 자동으로 빠진다
        ms_h = [m for m in ms if m in e.model.unique()]
        e = common_subset(e[e.model.isin(ms_h)])
        if e.empty:
            continue
        e["ae"] = (e.y_pred - e.y_true).abs()
        e["sae"] = e.ae / e.den.where(e.den > 0)
        for m, g in e.groupby("model"):
            per_series = g.groupby("series_id").sae.mean()
            gp = g[g.den > 0]
            row = dict(target=target, level=level, fold=fold, h=h, scope=scope, model=m,
                       n_series=g.series_id.nunique(), n_origins=g.origin.nunique(),
                       MAE=g.ae.mean(), MASE_mean=per_series.mean(), MASE_median=per_series.median(),
                       MASE_pooled=gp.ae.sum() / gp.den.sum() if len(gp) else np.nan)
            if target in D.COUNT_TARGETS:
                row["PoissonDev"] = pois_dev(g.y_true.values, g.y_pred.values).mean()
            metrics.append(row)
        # origin별 손실 (DM용): 계열 평균 척도 오차
        L = e.groupby(["model", "origin"]).sae.mean().unstack("model")
        loss_store[(target, level, fold, h, scope)] = L

met = pd.DataFrame(metrics)
met.to_csv(R / f"06_metrics{SUF}.csv", index=False, encoding="utf-8-sig")


# ---------------- DM-HLN ----------------
def dm_hln(d, h):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan, n
    dbar = d.mean()
    dc = d - dbar
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(0, min(h, n))]
    V = gam[0] + 2 * sum(gam[1:])
    if V <= 0:   # 직사각 가중이 음수면 Bartlett 가중
        V = gam[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gam[1:], 1))
    if V <= 0:
        return np.nan, np.nan, n
    dm = dbar / np.sqrt(V / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * k
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p, n


for (target, level, fold, h, scope), L in loss_store.items():
    pairs = [(m, "SeasonalNaive") for m in L.columns if m != "SeasonalNaive"] + \
            [(m, "Naive") for m in L.columns if m not in ("Naive", "SeasonalNaive")]
    for a, b in (("LightGBM+x", "LightGBM"), ("Ridge+x", "Ridge"), ("LightGBM", "Ridge"),
                 ("LightGBM+x·직접", "LightGBM·직접"), ("Ridge+x·직접", "Ridge·직접")):
        if a in L.columns and b in L.columns:
            pairs.append((a, b))
    for a, b in pairs:
        if b not in L.columns:
            continue
        d = (L[a] - L[b]).values
        stat, p, n = dm_hln(d, h)
        dms.append(dict(target=target, level=level, fold=fold, h=h, scope=scope, model=a, vs=b,
                        mean_loss_diff=np.nanmean(d), dm_hln=stat, p=p, n_origins=n))
dm = pd.DataFrame(dms)
dm.to_csv(R / f"06_dm{SUF}.csv", index=False, encoding="utf-8-sig")

# ---------------- top-K ----------------
ma3 = D.h_agg(D.search_wide("share", final=FINAL), 3, "mean").shift(3)   # 최근 3개월(t-2..t) 평균
ma3.columns = ma3.columns.astype(str)
lvl_pct = ma3.rank(axis=1, pct=True)
Sd = D.stock("dong", FINAL)
Sd.columns = Sd.columns.astype(str)
TK = {"search": "검색 비중 변화", "open": "개업 수", f"net_rate_{C.MAIN_PANEL}": "순증률", "net_rate_v_keep": "순증률"}
PARTIAL = {"SARIMA", "SARIMAX", "Croston", "TSB"}   # 일부 동에만 적용: 전체 동 순위 비교에서 뺀다
for (target, fold, h), e0 in ev[(ev.level == "dong") & ev.target.isin(TK)].groupby(["target", "fold", "h"]):
    if target != "search":
        e0 = e0[~e0.model.isin(PARTIAL)]
    for m, g in e0.groupby("model"):
        hits = {K: [] for K in C.TOPK}
        for o, gg in g.groupby("origin"):
            gg = gg.copy()
            if target == "search":
                base = gg.series_id.map(ma3.loc[o])
                gg = gg[gg.series_id.map(lvl_pct.loc[o]) >= C.TOPK_SEARCH_MIN_LEVEL_PCT]
                base = base.loc[gg.index]
                ps, ts = gg.y_pred / base - 1, gg.y_true / base - 1
            elif target.startswith("net_rate"):
                gg = gg[gg.series_id.map(Sd.loc[o]) >= C.TOPK_NETRATE_MIN_STOCK]
                ps, ts = gg.y_pred, gg.y_true
            else:
                ps, ts = gg.y_pred, gg.y_true
            if ps.round(9).nunique() <= 1:   # 모든 동을 같게 예측하면(검색 Naive의 변화 = 0) 순위가 없다
                continue
            for K in C.TOPK:
                top_p = set(gg.series_id[ps.rank(ascending=False, method="first") <= K])
                top_t = set(gg.series_id[ts.rank(ascending=False, method="first") <= K])
                hits[K].append(len(top_p & top_t) / K)
        if not hits[C.TOPK[0]]:
            continue
        r = dict(target=target, fold=fold, h=h, model=m, n_origins=len(hits[C.TOPK[0]]), n_dongs=len(gg))
        for K in C.TOPK:
            r[f"top{K}"] = np.mean(hits[K])
            r[f"random{K}"] = K / max(len(gg), 1)
        topk.append(r)
tk = pd.DataFrame(topk)
tk.to_csv(R / f"06_topk{SUF}.csv", index=False, encoding="utf-8-sig")

print(met[(met.h == 3)].sort_values(["target", "level", "scope", "fold", "MASE_mean"]).to_string())
log_run("06_evaluate", "MASE, PoissonDev, top-K, DM-HLN", rows=len(met), final=FINAL)


# ================= 보고서 =================
NAME = {"search": "검색 비중 (h개월 평균)", "open": "개업 수", "close_v_drop": "폐업 수 (일괄 폐업 제외)",
        "close_v_keep": "폐업 수 (일괄 폐업 포함)", "net_rate_v_drop": "순증률 (일괄 폐업 제외)",
        "net_rate_v_keep": "순증률 (일괄 폐업 포함)", "churn_v_drop": "회전율 (일괄 폐업 제외)",
        "churn_v_keep": "회전율 (일괄 폐업 포함)"}
LV = {"seoul": "서울", "gu": "구", "dong": "동"}
SC = {"all": "전체 동", "busy": "개폐업 많은 동", "zero": "0 많은 동"}


def fmt(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def dm_mark(target, level, fold, h, scope, model, vs="SeasonalNaive"):
    r = dm[(dm.target == target) & (dm.level == level) & (dm.fold == fold) & (dm.h == h) & (dm.scope == scope)
           & (dm.model == model) & (dm.vs == vs)]
    if r.empty or pd.isna(r.p.iloc[0]):
        return ""
    p, s = r.p.iloc[0], r.dm_hln.iloc[0]
    star = "**" if p < .05 else ("*" if p < .1 else "")
    return f"{'+' if s > 0 else '−'}{star}" if star else ""


def table(target, level, h, scope="all", cols=("MASE_mean", "MASE_median")):
    m = met[(met.target == target) & (met.level == level) & (met.h == h) & (met.scope == scope)]
    if m.empty:
        return []
    out = ["| 모델 | " + " | ".join(f"{f} {c.replace('MASE_', 'MASE ')}" for f in FOLDS for c in cols)
           + (" | " + " | ".join(f"{f} Poisson dev." for f in FOLDS) if target in D.COUNT_TARGETS else "") + " |",
           "|---|" + "---:|" * (len(FOLDS) * len(cols) + (len(FOLDS) if target in D.COUNT_TARGETS else 0))]
    order = m[m.fold == FOLDS[-1]].sort_values("MASE_mean").model.tolist()
    order += [x for x in m.model.unique() if x not in order]
    for mod in order:
        cells = []
        for f in FOLDS:
            r = m[(m.fold == f) & (m.model == mod)]
            for c in cols:
                v = fmt(r[c].iloc[0]) if len(r) else ""
                if c == "MASE_mean" and len(r):
                    v += f" {dm_mark(target, level, f, h, scope, mod)}".rstrip()
                cells.append(v)
        if target in D.COUNT_TARGETS:
            for f in FOLDS:
                r = m[(m.fold == f) & (m.model == mod)]
                cells.append(fmt(r.PoissonDev.iloc[0], 2) if len(r) else "")
        out.append(f"| {mod} | " + " | ".join(cells) + " |")
    n = m.iloc[0]
    out.append(f"\n계열 {int(n.n_series)}개 × origin {int(n.n_origins)}개 ({n.fold}). "
               "MASE 옆 표시: 계절 naive와 DM-HLN 검정, `−` 계절 naive보다 오차가 작음 / `+` 큼, ** p<0.05, * p<0.1.\n")
    return out


def topk_table(target, h):
    t = tk[(tk.target == target) & (tk.h == h)]
    if t.empty:
        return []
    out = ["| 모델 | " + " | ".join(f"{f} top{K}" for f in FOLDS for K in C.TOPK) + " |",
           "|---|" + "---:|" * (len(FOLDS) * len(C.TOPK))]
    order = t[t.fold == FOLDS[-1]].sort_values(f"top{C.TOPK[0]}", ascending=False).model.tolist()
    order += [x for x in t.model.unique() if x not in order]
    for mod in order:
        cells = []
        for f in FOLDS:
            r = t[(t.fold == f) & (t.model == mod)]
            cells += [fmt(r[f"top{K}"].iloc[0], 2) if len(r) else "" for K in C.TOPK]
        out.append(f"| {mod} | " + " | ".join(cells) + " |")
    r0 = t.iloc[0]
    out.append(f"\n대상 동 {int(r0.n_dongs)}개. 무작위로 뽑을 때 기대 적중률: "
               + ", ".join(f"top{K} {r0[f'random{K}']:.2f}" for K in C.TOPK) + "\n")
    return out


def bar(ax, target, level, h, scope, title):
    m = met[(met.target == target) & (met.level == level) & (met.h == h) & (met.scope == scope)]
    if m.empty:
        ax.set_visible(False)
        return
    order = m.groupby("model").MASE_mean.mean().sort_values().index
    x = np.arange(len(order))
    for i, f in enumerate(FOLDS):
        v = m[m.fold == f].set_index("model").MASE_mean.reindex(order)
        ax.barh(x + (i - .5) * .38, v.values, height=.38, label=f, color=["#2a6f97", "#e0a33b"][i % 2])
    ax.axvline(1, color="k", lw=.8, ls="--")
    ax.set_yticks(x)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MASE (1 = 계절 naive)")
    ax.legend(fontsize=7)


fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for ax, (t, lv, sc_, ti) in zip(axs.flat, [("search", "dong", "all", "검색 비중 (동, h=3)"),
                                           ("open", "dong", "all", "개업 수 (동, h=3)"),
                                           (f"close_{C.MAIN_PANEL}", "dong", "all", "폐업 수 (동, h=3, v_drop)"),
                                           (f"net_rate_{C.MAIN_PANEL}", "dong", "all", "순증률 (동, h=3, v_drop)")]):
    bar(ax, t, lv, C.MAIN_H, sc_, ti)
fig.tight_layout()
fig.savefig(R / f"06_mase_h3{SUF}.png", dpi=140)

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (t, ti) in zip(axs, [("search", "검색 비중 변화"), ("open", "개업 수"), (f"net_rate_{C.MAIN_PANEL}", "순증률")]):
    x = tk[(tk.target == t) & (tk.h == C.MAIN_H)]
    if x.empty:
        continue
    order = x.groupby("model").top20.mean().sort_values(ascending=False).index
    xi = np.arange(len(order))
    for i, f in enumerate(FOLDS):
        v = x[x.fold == f].set_index("model").top20.reindex(order)
        ax.bar(xi + (i - .5) * .38, v.values, width=.38, label=f, color=["#2a6f97", "#e0a33b"][i % 2])
    ax.axhline(x.random20.iloc[0], color="k", lw=.8, ls="--", label="무작위")
    ax.set_xticks(xi)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"{ti}: top-20 적중률 (h=3)", fontsize=10)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / f"06_topk_h3{SUF}.png", dpi=140)

L = ["# 예측 결과 표 (자동 생성)\n",
     f"> `src/forecast/06_evaluate.py{' --final' if FINAL else ''}`. fold: {', '.join(FOLDS)}. "
     "MASE = 계열별 (모델 MAE / 학습 구간 계절 naive MAE)의 평균·중앙값. 1보다 작으면 계절 naive보다 낫다.\n"]
for h in C.HORIZONS:
    L.append(f"## h = {h}\n")
    for target in ["search", "open", f"close_{C.MAIN_PANEL}", f"net_rate_{C.MAIN_PANEL}", f"churn_{C.MAIN_PANEL}",
                   "close_v_keep", "net_rate_v_keep"]:
        for level in ("dong", "gu", "seoul"):
            for scope in (["all", "busy", "zero"] if level == "dong" else ["all"]):
                tb = table(target, level, h, scope)
                if tb:
                    L.append(f"### {NAME[target]} · {LV[level]}" + (f" · {SC[scope]}" if level == "dong" and target != "search" else "") + "\n")
                    L += tb
    for target in ["search", "open", f"net_rate_{C.MAIN_PANEL}"]:
        tb = topk_table(target, h)
        if tb:
            L.append(f"### top-K 적중률 · {NAME[target]}\n")
            L += tb
(R / f"06_tables{SUF}.md").write_text("\n".join(L), encoding="utf-8")
print("보고서:", R / f"06_tables{SUF}.md")
