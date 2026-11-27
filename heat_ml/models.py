"""Fit and tune global Ridge and LightGBM using expanding windows."""

from itertools import product

import numpy as np
import pandas as pd

from .data import CONDITIONS, HORIZON, WEEK
from .festival import FESTIVAL_GROUPS
from .level import LEVEL_GROUPS
from .robust import ROBUST_GROUPS
from .semester import SEMESTER_GROUPS

RIDGE = [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)]
LGBM = [dict(num_leaves=n, min_data_in_leaf=m, learning_rate=r)
        for n, m, r in product((7, 15, 31), (50, 200), (0.03, 0.07))]
GRIDS = {"ridge": RIDGE, "lightgbm": LGBM}
FEATURE_GROUPS = {
    "lags": ("lag1", "lag2", "lag3", "lag4", "lag13", "lag52"),
    "means": ("mean4", "mean13"),
    "prior_year": ("target_lag52",),
    "fourier": ("sin1", "cos1", "sin2", "cos2", "sin3", "cos3"),
}
ABLATIONS = {
    "minimal": (),
    **{f"only_{group}": (group,) for group in FEATURE_GROUPS},
    "base": tuple(FEATURE_GROUPS),
    **{f"without_{group}": tuple(other for other in FEATURE_GROUPS if other != group)
       for group in FEATURE_GROUPS},
}
PHASE2_FEATURES = {
    "recent_lags": ("lag1", "lag2", "lag3", "lag4"),
    "long_lags": ("lag13", "lag52"),
    "lags_with_lag0": ("lag0", *FEATURE_GROUPS["lags"]),
    "mean4_only": ("mean4",),
    "mean13_only": ("mean13",),
    "no_seasonal_features": (),
    "base_lag0": ("lag0",),  # v2 baseline: every basic group plus the origin week.
    # v2 P5/F4: base_lag0 with festival-aligned prior-year lag and target value.
    "base_lag0_aligned": ("lag0", "lag1", "lag2", "lag3", "lag4", "lag13",
                          "lag52_aligned", "target_lag52_aligned"),
    # v2 P6: base_lag0 with clipped recent lags and means.
    "base_lag0_robust": ("rlag0", "rlag1", "rlag2", "rlag3", "rlag4", "lag13", "lag52", "rmean4", "rmean13"),
}
# Named groups usable after "+" in a condition; other names are taken as single columns.
# v3 simple groups, added one at a time after "minimal" (e.g. "minimal+lag0+recent_lags+means").
V3_GROUPS = {
    **FEATURE_GROUPS,
    "recent_lags": ("lag1", "lag2", "lag3", "lag4"),
    "long_lags": ("lag13", "lag52"),
}
EXTRA_GROUPS = {**V3_GROUPS, **FESTIVAL_GROUPS, **SEMESTER_GROUPS, **LEVEL_GROUPS, **ROBUST_GROUPS}
PHASE2_REPLACED_GROUPS = {
    "recent_lags": ("lags",), "long_lags": ("lags",), "lags_with_lag0": ("lags",),
    "mean4_only": ("means",), "mean13_only": ("means",),
    "no_seasonal_features": ("prior_year", "fourier"),
    "base_lag0_aligned": ("lags", "prior_year"),
    "base_lag0_robust": ("lags", "means"),
}


def wape(actual, predicted):
    denominator = np.abs(actual).sum()
    return float(100 * np.abs(actual - predicted).sum() / denominator) if denominator > 0 else np.nan


def restore(log_prediction):
    return np.maximum(np.expm1(log_prediction), 0)


def design(train, future, condition, model):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    base, *added = condition.split("+")
    groups = ABLATIONS.get(base, tuple(FEATURE_GROUPS))
    replaced = PHASE2_REPLACED_GROUPS.get(base, ())
    numeric = ["h", *(name for group in groups if group not in replaced for name in FEATURE_GROUPS[group]),
               *PHASE2_FEATURES.get(base, ()),
               *CONDITIONS.get(base, ()),
               *(column for name in added for column in EXTRA_GROUPS.get(name, (name,)))]
    if model == "ridge":
        preprocessing = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                      keep_empty_features=True), StandardScaler()), numeric),
            ("unit", OneHotEncoder(handle_unknown="ignore"), ["unit"]),
        ])
        return preprocessing.fit_transform(train), preprocessing.transform(future)
    # LightGBM learns missing branches from training data. The category vocabulary is split-local.
    categories = pd.Index(train.unit.unique())
    return tuple(frame[numeric].assign(dong=pd.Categorical(frame.unit, categories=categories))
                 for frame in (train, future))


def fit_predict(model, params, train_x, train_y, future_x, jobs, iterations, validation=None,
                patience=100, train_offset=None, future_offset=None, offset_space="raw"):
    if train_offset is None:
        fit_y = np.log1p(train_y)
        valid_y = None if validation is None else np.log1p(validation)
        decode = restore
    elif offset_space == "log":
        # Fit log1p(y) − offset (v2 F8b: the dong's log share when the offset is log Seoul).
        fit_y = np.log1p(train_y) - train_offset
        valid_y = None if validation is None else np.log1p(validation) - future_offset

        def decode(predicted):
            return restore(predicted + future_offset)
    else:
        fit_y = train_y - train_offset
        valid_y = None if validation is None else validation - future_offset

        def decode(predicted):
            return np.maximum(predicted + future_offset, 0)

    if model == "ridge":
        from sklearn.linear_model import Ridge
        ridge = Ridge(**params, solver="lsqr").fit(train_x, fit_y)
        return decode(ridge.predict(future_x)), None

    import lightgbm as lgb

    booster = lgb.LGBMRegressor(
        **params, n_estimators=iterations, random_state=42, n_jobs=jobs,
        verbosity=-1, deterministic=True, force_col_wise=True, metric="None",
    )
    if validation is None:
        booster.fit(train_x, fit_y)
    else:
        def original_scale_wape(actual, predicted):
            return "wape", wape(validation, decode(predicted)), False

        booster.fit(train_x, fit_y, eval_set=[(future_x, valid_y)],
                    eval_metric=original_scale_wape, callbacks=[lgb.early_stopping(patience, verbose=False)])
    return decode(booster.predict(future_x)), booster.best_iteration_ or iterations


def tune(panel, cutoff, model, condition, jobs, quick=False, staged=False, offset_column=None,
         require_seasonal=False, offset_space="raw"):
    grid = GRIDS[model][:1] if quick else GRIDS[model]
    iterations = 30 if quick else 2000
    results = {}
    folds = []
    # Two adjacent, nonoverlapping 13-week validation blocks, ending at cutoff.
    for inner in (cutoff - 2 * HORIZON * WEEK, cutoff - HORIZON * WEEK):
        train = panel.training(inner)
        valid = panel.examples([inner]).dropna(subset=["y"])
        if require_seasonal:
            valid = valid.dropna(subset=["seasonal"])
        if valid.empty or valid.week_end.max() > cutoff or valid.y.sum() <= 0:
            raise ValueError(f"Invalid nested validation at {inner.date()} for {cutoff.date()}")
        x, vx = design(train, valid, condition, model)
        offsets = ({} if offset_column is None else
                   dict(train_offset=train[offset_column].to_numpy(), future_offset=valid[offset_column].to_numpy(),
                        offset_space=offset_space))
        folds.append((inner, train.y, valid.y.to_numpy(), x, vx, offsets))

    def evaluate(params):
        key = tuple(params.items())
        if key in results:
            return results[key][1]
        scores = []
        for inner, train_y, valid_y, x, vx, offsets in folds:
            pred, steps = fit_predict(model, params, x, train_y, vx, jobs, iterations,
                                      validation=valid_y, patience=5 if quick else 100, **offsets)
            scores.append((inner.date().isoformat(), wape(valid_y, pred), steps))
        mean = float(np.mean([score[1] for score in scores]))
        results[key] = (scores, mean)
        return mean

    if staged and model == "lightgbm" and not quick:
        best = dict(num_leaves=15, min_data_in_leaf=200, learning_rate=0.03)
        for name, values in (("num_leaves", (7, 15, 31)),
                             ("min_data_in_leaf", (50, 200)),
                             ("learning_rate", (0.03, 0.07))):
            best = min((dict(best, **{name: value}) for value in values), key=evaluate)
    else:
        best = min(grid, key=evaluate)
    best_key = tuple(best.items())
    steps = None
    if model == "lightgbm":
        steps = int(np.median([score[2] for score in results[best_key][0]]))
    audit = [dict(params=dict(key), folds=scores, mean_wape=mean, selected=key == best_key)
             for key, (scores, mean) in results.items()]
    return best, steps, audit


def forecast(panel, origin, model, condition, jobs, quick=False, staged=False, offset_column=None,
             require_seasonal=False, offset_space="raw"):
    params, steps, audit = tune(panel, origin, model, condition, jobs, quick, staged, offset_column,
                                require_seasonal, offset_space)
    train = panel.training(origin)
    future = panel.examples([origin])
    x, vx = design(train, future, condition, model)
    offsets = ({} if offset_column is None else
               dict(train_offset=train[offset_column].to_numpy(), future_offset=future[offset_column].to_numpy(),
                    offset_space=offset_space))
    prediction, _ = fit_predict(model, params, x, train.y, vx, jobs, steps, **offsets)
    return prediction, {"params": params, "iterations": steps, "grid": audit, "training_rows": len(train)}

