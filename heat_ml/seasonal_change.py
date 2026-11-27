"""Backtest direct mobility and additive year-over-year change with shared features."""

import argparse
from pathlib import Path

import pandas as pd

from .data import load_target
from .experiment import csv, save_json
from .features import Panel, WEEK
from .models import GRIDS, forecast
from .phase2 import PACKAGE, run_comparison

FEATURES = "only_means+target_lag52+recent_yoy"
TARGET_OFFSETS = {"direct_raw": "zero_offset", "seasonal_change": "seasonal", "direct_log": None}


class SeasonalChangePanel(Panel):
    def examples(self, origins, label_cutoff=None):
        frame = super().examples(origins, label_cutoff)
        changes = []
        for origin in origins:
            history = self.corrected(origin)
            recent = history.tail(4)
            previous = history.reindex(recent.index - 52 * WEEK)
            paired_mean = (recent - previous.to_numpy()).mean()
            changes.extend(zip([origin] * len(self.units), self.units, paired_mean))
        recent_changes = pd.DataFrame(changes, columns=["origin", "unit", "recent_yoy"])
        frame = frame.merge(recent_changes, on=["origin", "unit"], how="left", validate="many_to_one")
        frame["zero_offset"] = 0.0
        if label_cutoff is not None:
            return frame.dropna(subset=["seasonal"])
        return frame


def predict(panel, origin, model, condition, jobs, staged):
    offset = TARGET_OFFSETS[condition]
    return forecast(panel, origin, model, FEATURES, jobs, staged=staged, offset_column=offset,
                    require_seasonal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=PACKAGE / "target_2023_2025.csv")
    parser.add_argument("--reference", type=Path, default=PACKAGE / "run_phase1/outer_predictions.csv")
    parser.add_argument("--out", type=Path, default=PACKAGE / "run_seasonal_change")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=False)
    save_json({"features": ["unit", "h", "mean4", "mean13", "target_lag52", "recent_yoy"],
               "targets": list(TARGET_OFFSETS), "requires_prior_year_for_training_and_tuning": True},
              out / "setup.json")
    panel = SeasonalChangePanel(load_target(args.target), {})
    candidates = [(model, condition, False) for model in GRIDS for condition in TARGET_OFFSETS]
    mean = run_comparison(panel, args.reference, out, args.jobs, candidates, predict=predict)
    csv(mean, out / "mean_wape.csv")


if __name__ == "__main__":
    main()
