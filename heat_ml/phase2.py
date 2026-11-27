"""Compare detailed feature subsets on the matched phase-one backtest rows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ORIGINS, load_target
from .experiment import CLEAN_ORIGINS, csv, save_json
from .features import Panel
from .models import GRIDS, PHASE2_FEATURES, forecast, wape

PACKAGE = Path(__file__).resolve().parent
KEYS = ["origin", "unit", "week_end"]
GROUPS = {
    "lags": ("without_lags", "recent_lags", "long_lags", "base", "lags_with_lag0"),
    "means": ("without_means", "mean4_only", "mean13_only", "base"),
    "seasonal": ("no_seasonal_features", "without_prior_year", "without_fourier", "base"),
}


def run(target, reference, out, jobs):
    conditions = tuple(dict.fromkeys(condition for group in GROUPS.values() for condition in group))
    candidates = [(model, condition, condition not in PHASE2_FEATURES)
                  for condition in conditions for model in GRIDS]
    mean = run_comparison(Panel(target, {}), reference, out, jobs, candidates)
    table = pd.DataFrame([(group, condition) for group, items in GROUPS.items() for condition in items],
                         columns=["feature_group", "condition"]).merge(mean, on="condition")
    csv(table, out / "mean_wape.csv")


def run_comparison(panel, reference, out, jobs, candidates, predict=forecast):
    reused = [f"{model}|{condition}|standalone" for model, condition, reuse in candidates if reuse]
    previous = pd.read_csv(reference, usecols=[*KEYS, "y", "matched", *reused], dtype={"unit": str},
                           parse_dates=["origin", "week_end"]).set_index(KEYS, verify_integrity=True)
    if set(previous.index.get_level_values("origin")) != set(ORIGINS):
        raise ValueError("Phase-one predictions must cover the four original forecast dates.")

    all_predictions, all_scores, tuning = [], [], {}
    for origin in ORIGINS:
        print(f"Backtest {origin.date()}", flush=True)
        frame = panel.examples([origin])
        prior = previous.reindex(pd.MultiIndex.from_frame(frame[KEYS]))
        if prior.matched.isna().any() or not np.allclose(prior.y, frame.y, equal_nan=True):
            raise ValueError(f"Phase-one target rows differ at {origin.date()}.")
        matched = prior.matched.to_numpy(dtype=bool)
        if not matched.any():
            raise ValueError(f"No matched phase-one rows at {origin.date()}.")
        wide = frame.loc[matched, [*KEYS, "h", "y"]].copy()
        tuning[str(origin.date())] = {}
        for model, condition, reuse in candidates:
            label = f"{model}|{condition}"
            if reuse:
                values = prior[f"{label}|standalone"].to_numpy()
            else:
                print(f"  {model} {condition}", flush=True)
                values, audit = predict(panel, origin, model, condition, jobs, staged=True)
                tuning[str(origin.date())][label] = audit
            wide[label] = values[matched]
            if not np.isfinite(wide[label]).all():
                raise ValueError(f"Missing {label} predictions on the reference evaluation rows.")
            for low, high in ((1, 13), (1, 4), (5, 8), (9, 13)):
                block = wide.loc[wide.h.between(low, high)]
                all_scores.append({"origin": origin, "model": model, "condition": condition,
                                   "horizon": f"{low}-{high}", "rows": len(block),
                                   "units": block.unit.nunique(), "wape": wape(block.y, block[label])})
        all_predictions.append(wide)
        csv(pd.DataFrame(all_scores), out / "outer_wape.csv")
        save_json(tuning, out / "tuning.json")
        print(f"Finished {origin.date()} ({len(wide)} matched rows)", flush=True)

    scores = pd.DataFrame(all_scores)
    keys = ["model", "condition", "horizon"]
    mean = scores.groupby(keys).wape.mean().rename("mean4").to_frame()
    mean["mean2"] = scores.loc[scores.origin.isin(CLEAN_ORIGINS)].groupby(keys).wape.mean()
    csv(pd.concat(all_predictions), out / "outer_predictions.csv")
    return mean.reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=PACKAGE / "target_2023_2025.csv")
    parser.add_argument("--reference", type=Path, default=PACKAGE / "run_phase1/outer_predictions.csv")
    parser.add_argument("--out", type=Path, default=PACKAGE / "run_phase2")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=False)
    run(load_target(args.target), args.reference, out, args.jobs)
