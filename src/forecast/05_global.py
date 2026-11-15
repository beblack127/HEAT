"""전역 모델: LightGBM (개수는 Poisson), Ridge (같은 피처). direct multi-horizon, fold당 한 번 학습.

대상: search, open, close_v_drop, close_v_keep, net_rate_v_drop, net_rate_v_keep (동 수준)
  search    y = log(t+1..t+h 평균 비중) − log(최근 3개월 평균 비중) -> 예측 비중 = 최근 3개월 평균 × exp(ŷ)
  개수      LightGBM Poisson(원래 개수) / Ridge log1p(개수)
  순증률    LightGBM L2 / Ridge (원값)
피처 그룹: base, base+cross (검색 모델에 F&B, 개폐업 모델에 검색). 이름에 '+x'가 붙은 모델이 교차 피처 버전.
학습 표본: origin TRAIN_START ~ (학습 끝 − h). early stopping·Ridge alpha는 학습 구간 마지막 3개월(실제값 끝 기준) 검증.
구·서울 값은 동 예측의 합(bottom-up)으로 만든다.
출력: results/forecast/pred_global.parquet, reports/forecast/05_feature_importance.csv
"""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import C, Timer, log_run, mrange, origins, shift_m, train_end
from features import Builder

warnings.filterwarnings("ignore")
FINAL = "--final" in sys.argv
FOLDS = ["FINAL"] if FINAL else list(C.FOLDS)
TARGETS = ["search", "open", "close_v_drop", "close_v_keep", "net_rate_v_drop", "net_rate_v_keep"]
GROUPS = {"base": "", "base+cross": "+x"}
NON_FEAT = {"origin", "series_id", "y", "base", "S_t", "gu"}


def fit_lgb(Xtr, ytr, Xva, yva, obj, feats):
    p = dict(C.LGB_PARAMS)
    n = p.pop("n_estimators")
    p["objective"] = obj
    ds = lgb.Dataset(Xtr[feats], ytr, categorical_feature=["gu_cat"], free_raw_data=False)
    dv = lgb.Dataset(Xva[feats], yva, reference=ds, categorical_feature=["gu_cat"])
    m = lgb.train(p, ds, n, valid_sets=[dv], callbacks=[lgb.early_stopping(C.EARLY_STOP_ROUNDS, verbose=False)])
    return max(m.best_iteration, 10)


def train_lgb(X, y, obj, feats, n_iter):
    p = dict(C.LGB_PARAMS)
    p.pop("n_estimators")
    p["objective"] = obj
    return lgb.train(p, lgb.Dataset(X[feats], y, categorical_feature=["gu_cat"]), n_iter)


def ridge_pipe(num, alpha):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["gu", "cal_month"]),
    ])
    return make_pipeline(pre, Ridge(alpha=alpha, random_state=C.SEED))


def clip_feats(X, num, lo, hi):
    X = X.copy()
    X[num] = X[num].clip(lo, hi, axis=1)
    return X


def main():
    rows, imps = [], []
    for fold in FOLDS:
        B = Builder(fold, FINAL)
        tend = train_end(fold)
        for target in TARGETS:
            for h in C.HORIZONS:
                tr_or = mrange(C.TRAIN_START, shift_m(tend, -h))
                va_or = tr_or[-C.EARLY_STOP_MONTHS:]
                te_or = origins(fold, h)
                for gname, suf in GROUPS.items():
                    with Timer(f"{fold} {target} h={h} {gname}"):
                        X = B.table(target, h, tr_or + te_or, gname)
                        X["gu_cat"] = X.gu.astype("category")
                        feats = [c for c in X.columns if c not in NON_FEAT]
                        num = [c for c in feats if c not in ("gu_cat", "cal_month")]
                        tr = X[X.origin.isin(tr_or) & X.y.notna()]
                        te = X[X.origin.isin(te_or)].copy()
                        fit, va = tr[~tr.origin.isin(va_or)], tr[tr.origin.isin(va_or)]
                        is_count = target in ("open", "close_v_drop", "close_v_keep")
                        obj = "poisson" if is_count else "regression"
                        # ---- LightGBM ----
                        n_it = fit_lgb(fit, fit.y, va, va.y, obj, feats)
                        m = train_lgb(tr, tr.y, obj, feats, n_it)
                        p_lgb = m.predict(te[feats])
                        imps.append(pd.DataFrame({"feature": feats, "gain": m.feature_importance("gain"),
                                                  "fold": fold, "target": target, "h": h, "group": gname}))
                        # ---- Ridge (같은 피처) ----
                        lo, hi = fit[num].quantile(.01), fit[num].quantile(.99)   # 학습 구간으로 정한 절단
                        ry = (lambda v: np.log1p(v)) if is_count else (lambda v: v)
                        rinv = (lambda v: np.expm1(v).clip(0)) if is_count else (lambda v: v)
                        rnum = num + ["cal_month"]
                        rcols = num + ["gu", "cal_month"]
                        best, best_a = np.inf, None
                        for a in C.RIDGE_ALPHAS:
                            pr = ridge_pipe(num, a).fit(clip_feats(fit, num, lo, hi)[rcols], ry(fit.y))
                            e = np.mean((pr.predict(clip_feats(va, num, lo, hi)[rcols]) - ry(va.y)) ** 2)
                            if e < best:
                                best, best_a = e, a
                        lo, hi = tr[num].quantile(.01), tr[num].quantile(.99)
                        pr = ridge_pipe(num, best_a).fit(clip_feats(tr, num, lo, hi)[rcols], ry(tr.y))
                        p_rdg = rinv(pr.predict(clip_feats(te, num, lo, hi)[rcols]))
                        for name, p in (("LightGBM", p_lgb), ("Ridge", p_rdg)):
                            if target == "search":
                                yp = te.base.values * np.exp(p)
                            else:
                                yp = p
                            rows.append(pd.DataFrame({"series_id": te.series_id.values, "origin": te.origin.values,
                                                      "y_pred": yp, "model": name + suf, "target": target,
                                                      "level": "dong", "fold": fold, "h": h,
                                                      "n_iter": n_it if name == "LightGBM" else np.nan,
                                                      "alpha": best_a if name == "Ridge" else np.nan}))
    out = pd.concat(rows, ignore_index=True)
    # 구·서울: 동 개수 예측의 합
    gu = Builder(FOLDS[0], FINAL).gu
    cnt = out[out.target.isin(["open", "close_v_drop", "close_v_keep"])].copy()
    cnt["gu"] = cnt.series_id.map(gu)
    k = ["target", "fold", "h", "model", "origin"]
    g = cnt.groupby(k + ["gu"]).y_pred.sum().reset_index().rename(columns={"gu": "series_id"}).assign(level="gu")
    s = cnt.groupby(k).y_pred.sum().reset_index().assign(series_id="서울", level="seoul")
    out = pd.concat([out, g, s], ignore_index=True)
    fn = "pred_global_final.parquet" if FINAL else "pred_global.parquet"
    out.to_parquet(C.RESULTS / fn, index=False)
    imp = pd.concat(imps)
    imp.to_csv(C.REPORTS / ("05_feature_importance_final.csv" if FINAL else "05_feature_importance.csv"),
               index=False, encoding="utf-8-sig")
    print(out.groupby(["target", "level", "fold", "h", "model"]).size().unstack("model").head(40))
    log_run("05_global", "LightGBM, Ridge (base, base+cross)", rows=len(out), final=FINAL)


if __name__ == "__main__":
    main()
