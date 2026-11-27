"""Re-evaluate the HEAT corrected seasonal and ARIMA blends at each origin."""

from itertools import product
import warnings

import numpy as np
import pandas as pd

from .data import HORIZON, WEEK
from .features import fourier

KEYS = ("seasonal", "arima_blend", "arima_fourier_blend")


def best_arima(values, future, exog=None, future_exog=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best = None
    best_aic = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, d, q in product(range(3), range(2), range(2)):
            try:
                fit = SARIMAX(values, exog=exog, order=(p, d, q),
                              trend="n" if d else "c").fit(disp=False, maxiter=200)
                if fit.aic < best_aic:
                    prediction = np.asarray(fit.get_forecast(len(future), exog=future_exog).predicted_mean)
                    if np.isfinite(prediction).all():
                        best_aic, best = fit.aic, prediction
            except (ValueError, np.linalg.LinAlgError, RuntimeError, OverflowError):
                continue
    return np.full(len(future), np.nan) if best is None else np.expm1(best)


def one_unit(history, future, seasonal, quick):
    values = np.log1p(history.dropna())
    if len(values) < 60:
        return np.full((2, len(future)), np.nan)
    # The smoke path checks output plumbing without running the expensive ARIMA grid.
    if quick:
        return np.vstack([seasonal, seasonal])
    exog = pd.DataFrame(fourier(values.index)).to_numpy()
    future_exog = pd.DataFrame(fourier(future)).to_numpy()
    plain = best_arima(values.to_numpy(), future)
    fourier_prediction = best_arima(values.to_numpy(), future, exog, future_exog)
    return (np.vstack([plain, fourier_prediction]) + seasonal) / 2


def baselines(panel, origin, frame, units, jobs, quick=False):
    from joblib import Parallel, delayed

    history = panel.corrected(origin)
    future = pd.date_range(origin + WEEK, periods=HORIZON, freq="W-SUN")
    seasonal = history.reindex(future - 52 * WEEK)
    members = sorted(units)
    results = Parallel(n_jobs=jobs)(
        delayed(one_unit)(history[unit], future, seasonal[unit].to_numpy(), quick)
        for unit in members
    )
    predictions = frame[["unit", "week_end", "seasonal"]].copy()
    for idx, key in enumerate(KEYS[1:]):
        index = pd.MultiIndex.from_product([members, future], names=["unit", "week_end"])
        values = pd.Series(np.concatenate([result[idx] for result in results]), index=index)
        predictions[key] = values.reindex(pd.MultiIndex.from_frame(frame[["unit", "week_end"]])).to_numpy()
    return predictions[list(KEYS)]

