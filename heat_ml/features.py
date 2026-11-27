"""Construct direct multi-horizon examples on a Sunday calendar."""

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import FINAL, HORIZON, START, WEEK

ANOMALY_START = pd.Timestamp("2024-08-01")
ANOMALY_END = pd.Timestamp("2024-10-31")


def fourier(dates):
    phase = np.asarray((dates - pd.Timestamp("2023-01-01")) / WEEK) * (2 * np.pi * 7 / 365.25)
    return {f"{f.__name__}{k}": f(k * phase) for k in range(1, 4) for f in (np.sin, np.cos)}


class Panel:
    # v2 sets this so a 4- or 13-week mean needs that many observed weeks (earlier runs did not).
    full_window_means = False

    def __init__(self, target, covariates):
        train = target.loc[target.week_end.le(FINAL)]
        self.weeks = pd.date_range(START, FINAL + HORIZON * WEEK, freq="W-SUN")
        self.raw = train.pivot(index="week_end", columns="unit", values="y").reindex(self.weeks)
        self.seoul = train.groupby("week_end").seoul_y.first().reindex(self.weeks)
        self.units = self.raw.columns.to_numpy()
        self.covariates = covariates

    @lru_cache(maxsize=256)
    def factor(self, cutoff):
        city = self.seoul.loc[:cutoff]
        anomaly = city.loc[ANOMALY_START:ANOMALY_END]
        if anomaly.empty:
            return 1.0
        ratio = city.loc["2024-01":"2024-07"].mean() / city.loc["2023-01":"2023-07"].mean()
        pair = pd.concat([anomaly, city.shift(52).reindex(anomaly.index) * ratio], axis=1).dropna()
        if pair.empty or pair.iloc[:, 1].sum() <= 0 or pair.iloc[:, 0].sum() <= 0:
            raise ValueError(f"Cannot estimate anomaly factor as of {cutoff.date()}")
        return float(pair.iloc[:, 0].sum() / pair.iloc[:, 1].sum())

    def corrected(self, cutoff):
        values = self.raw.loc[:cutoff].copy()
        values.loc[ANOMALY_START:ANOMALY_END] /= self.factor(cutoff)
        return values

    def examples(self, origins, label_cutoff=None):
        sizes = len(self.units)
        if label_cutoff is not None:
            labels = self.corrected(label_cutoff)
        frames = []
        for origin in origins:
            history = self.corrected(origin)
            targets = pd.date_range(origin + WEEK, periods=HORIZON, freq="W-SUN")
            frame = pd.DataFrame({
                "origin": origin, "week_end": targets.repeat(sizes),
                "unit": np.tile(self.units, HORIZON), "h": np.repeat(np.arange(1, HORIZON + 1), sizes),
            })
            log = np.log1p(history)
            for lag in (0, 1, 2, 3, 4, 13, 52):
                frame[f"lag{lag}"] = np.tile(log.reindex([origin - lag * WEEK]).iloc[0], HORIZON)
            for width in (4, 13):
                window = history.tail(width)
                mean = np.log1p(window.mean())
                if self.full_window_means:
                    mean = mean.where(window.notna().sum().eq(width))
                frame[f"mean{width}"] = np.tile(mean, HORIZON)
            frame["seasonal"] = history.reindex(targets - 52 * WEEK).to_numpy().ravel()
            frame["target_lag52"] = np.log1p(frame.seasonal)
            for name, values in fourier(targets).items():
                frame[name] = np.repeat(values, sizes)
            for name, values in self.covariates.items():
                periods = (targets - 13 * WEEK if name == "subway"
                           else (targets.to_period("M") - 4).to_timestamp("M"))
                key = pd.MultiIndex.from_arrays([frame.unit, periods.repeat(sizes)])
                observed = values.reindex(key)
                frame[name] = observed[name].where(observed[f"{name}_available"].le(origin)).to_numpy()
            target = self.raw if label_cutoff is None else labels
            frame["y"] = target.reindex(targets).to_numpy().ravel()
            frames.append(frame if label_cutoff is None else frame.dropna(subset=["y"]))
        return pd.concat(frames, ignore_index=True)

    @lru_cache(maxsize=3)
    def training(self, cutoff):
        frame = self.examples(self.weeks[self.weeks < cutoff], cutoff)
        if frame.empty:
            raise ValueError(f"No training labels through {cutoff.date()}")
        return frame

