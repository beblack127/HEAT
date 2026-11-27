"""Backtest external search and restaurant features against standalone ML baselines."""

import argparse
from pathlib import Path

from .data import FNB, ORIGINS, SEARCH, load_covariates, load_target
from .experiment import csv, save_json
from .features import Panel
from .phase2 import PACKAGE, run_comparison

BASES = {"ridge": "only_fourier", "lightgbm": "only_prior_year"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=PACKAGE / "target_2023_2025.csv")
    parser.add_argument("--reference", type=Path, default=PACKAGE / "run_phase1/outer_predictions.csv")
    parser.add_argument("--out", type=Path, default=PACKAGE / "run_phase3")
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--assume-period-end", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.releases is None and not args.assume_period_end:
        parser.error("Provide release dates or explicitly select a retrospective month-end assumption.")
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=False)

    covariates = load_covariates({"search": SEARCH, "fnb": FNB}, args.releases,
                                assume_period_end=args.assume_period_end)
    panel = Panel(load_target(args.target), covariates)
    coverage = []
    for origin in ORIGINS:
        frame = panel.examples([origin])
        for name in ("search", "fnb"):
            if name not in frame or frame[name].notna().sum() == 0:
                raise ValueError(f"No available {name} inputs at {origin.date()}.")
            coverage.append({"origin": str(origin.date()), "feature": name, "rows": len(frame),
                             "observed": int(frame[name].notna().sum())})
    save_json({"availability": "retrospective_month_end_assumption" if args.releases is None else "release_dates",
               "monthly_lag": 4, "base_conditions": BASES, "input_coverage": coverage}, out / "setup.json")
    candidates = [(model, base + suffix, suffix == "") for model, base in BASES.items()
                  for suffix in ("", "+search", "+fnb", "+search+fnb")]
    table = run_comparison(panel, args.reference, out, args.jobs, candidates)
    table["external"] = table.condition.map(lambda name: name.partition("+")[2] or "none")
    table["availability"] = "retrospective_month_end_assumption" if args.releases is None else "release_dates"
    csv(table, out / "mean_wape.csv")


if __name__ == "__main__":
    main()
