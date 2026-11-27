"""Deterministic synthetic checks; no real WAPE is computed here."""

import numpy as np
import pandas as pd

from .data import CONDITIONS, FINAL, ORIGINS, START, WEEK
from .experiment import run
from .features import ANOMALY_END, ANOMALY_START, Panel
from .models import wape


def check(out=None):
    weeks = pd.date_range(START, FINAL + 13 * WEEK, freq="W-SUN")
    target = pd.MultiIndex.from_product([weeks, ["a", "b", "c", "d"]],
                                        names=["week_end", "unit"]).to_frame(index=False)
    target["y"] = np.tile([100., 150., 200., 250.], len(weeks))
    target.loc[target.week_end.between(ANOMALY_START, ANOMALY_END), "y"] *= 0.5
    target["seoul_y"] = target.groupby("week_end").y.transform("sum")
    month = pd.period_range("2022-01", "2025-12", freq="M").to_timestamp("M")
    covariates = {}
    for name in ("search", "fnb", "subway"):
        periods = weeks if name == "subway" else month
        idx = pd.MultiIndex.from_product([target.unit.unique(), periods], names=["unit", "period"])
        covariates[name] = pd.DataFrame({
            name: np.arange(len(idx), dtype=float),
            f"{name}_available": idx.get_level_values("period") + pd.Timedelta(days=1),
        }, index=idx)
    panel = Panel(target, covariates)
    origin = ORIGINS[0]
    assert panel.factor(origin) == 0.5
    assert panel.factor(pd.Timestamp("2024-07-28")) == 1
    training = panel.training(origin)
    assert training.week_end.max() == origin
    assert training.week_end.gt(training.origin).all()
    assert training.loc[training.week_end.between(ANOMALY_START, origin), "y"].min() == 100
    future = panel.examples([FINAL])
    assert future.y.isna().all()
    assert future.h.nunique() == 13 and future.week_end.max() == pd.Timestamp("2026-03-29")
    assert future.loc[future.h.eq(13), "subway"].isna().all()
    assert future.loc[future.h.eq(12), "subway"].notna().all()
    assert wape(np.array([100, 200]), np.array([90, 180])) == 10
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
        run(target, covariates, None, out, tuple(CONDITIONS), 1, quick=True)
    print("Synthetic smoke checks passed.")

