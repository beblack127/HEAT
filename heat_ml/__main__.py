"""Command-line entry point for the HEAT 2026 experiment."""

import argparse
from importlib.util import find_spec
from pathlib import Path

from .data import (COMMON, CONDITIONS, FINAL, FNB, ROOT, SEARCH, SUBWAY,
                   common_units, load_covariates, load_target, prepare_mobility, read)
from .experiment import csv, run, score_final
from .models import ABLATIONS

PACKAGE = Path(__file__).resolve().parent
TARGET = PACKAGE / "target_2023_2025.csv"
HEAT_TARGET = ROOT / "HEAT/data/interim/gt/target_weekly.parquet"


def output_path(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(PACKAGE) or resolved == PACKAGE:
        raise ValueError("Output must be inside heat_ml/.")
    return resolved


def available(path):
    return path if path.is_file() else None


def require_models(include_arima=True):
    names = ("sklearn", "lightgbm", "joblib", "statsmodels") if include_arima else (
        "sklearn", "lightgbm", "joblib",
    )
    missing = [name for name in names if find_spec(name) is None]
    if missing:
        raise ImportError("Missing model dependencies: " + ", ".join(missing)
                          + ". Install with python -m pip install -r heat_ml/requirements.txt")


def cli():
    parser = argparse.ArgumentParser(prog="python -m heat_ml")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Aggregate original OA-22298 mobility archives")
    prepare.add_argument("--mobility-dir", type=Path, default=ROOT / "data")
    prepare.add_argument("--first", default="2023-01-02")
    prepare.add_argument("--last", default=str(FINAL.date()))
    prepare.add_argument("--output", type=Path, default=TARGET)
    experiment = commands.add_parser("run", help="Backtest and freeze the 13-week forecast")
    experiment.add_argument("--target", type=Path)
    experiment.add_argument("--truth", type=Path, help="Score this weekly truth after freezing the forecast")
    experiment.add_argument("--common", type=Path)
    experiment.add_argument("--historical-rows", type=Path, help="Exact HEAT t7 origin/unit/week_end keys")
    experiment.add_argument("--search", type=Path, default=SEARCH)
    experiment.add_argument("--fnb", type=Path, default=FNB)
    experiment.add_argument("--subway", type=Path, default=SUBWAY)
    experiment.add_argument("--releases", type=Path)
    experiment.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    experiment.add_argument("--allow-partial", action="store_true")
    experiment.add_argument("--phase1", action="store_true", help="Run the 10 basic feature ablations")
    experiment.add_argument("--out", required=True, type=Path)
    experiment.add_argument("--jobs", type=int, default=1)
    score = commands.add_parser("score", help="Evaluate a frozen forecast against raw 2026 truth")
    score.add_argument("--forecast", required=True, type=Path)
    score.add_argument("--truth", required=True, type=Path)
    score.add_argument("--out", required=True, type=Path)
    smoke = commands.add_parser("smoke", help="Run synthetic cutoff and scoring checks")
    smoke.add_argument("--full", action="store_true", help="Also run reduced Ridge and LightGBM grids")
    smoke.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            out = output_path(args.output)
            if out.exists():
                raise ValueError(f"Output already exists: {out}")
            weekly = prepare_mobility(args.mobility_dir, args.first, args.last)
            out.parent.mkdir(parents=True, exist_ok=True)
            csv(weekly, out)
            print(f"Wrote {len(weekly)} weekly rows across {weekly.unit.nunique()} units to {out.name}")
        elif args.command == "run":
            if args.phase1 and args.truth is not None:
                raise ValueError("Phase 1 is a backtest. Score final truth after a final forecast run.")
            target_path = args.target or available(HEAT_TARGET) or available(TARGET)
            common_path = args.common or available(COMMON)
            missing = []
            if target_path is None:
                missing.append("HEAT-format weekly target; run 'python -m heat_ml prepare'")
            if common_path is None and not args.allow_partial:
                missing.append("HEAT/results/gt/unit_metrics.parquet or --common CSV of exact membership")
            conditions = tuple(ABLATIONS) if args.phase1 else tuple(
                condition for condition in CONDITIONS if condition in args.conditions)
            if not args.phase1 and conditions != tuple(CONDITIONS) and not args.allow_partial:
                raise ValueError("A restricted condition set requires --allow-partial.")
            needed = set().union(*(CONDITIONS.get(condition, ()) for condition in conditions))
            if needed and args.releases is None:
                missing.append("--releases CSV supplying historical availability by feature and period")
            paths = {name: getattr(args, name) for name in needed}
            missing.extend(f"{name}: {path}" for name, path in sorted(paths.items()) if not path.is_file())
            if missing:
                raise ValueError("Missing required inputs:\n  " + "\n  ".join(missing))
            if args.jobs == 0:
                raise ValueError("--jobs must be nonzero.")
            require_models()
            target = load_target(target_path)
            members = common_units(common_path, target.loc[target.week_end.le(FINAL), "unit"].unique())
            covariates = load_covariates(paths, args.releases)
            historical = read(args.historical_rows) if args.historical_rows is not None else None
            out = output_path(args.out)
            # A new directory prevents an incomplete rerun from replacing a frozen forecast.
            out.mkdir(parents=True, exist_ok=False)
            run(target, covariates, members, out, conditions, args.jobs, historical=historical,
                phase1=args.phase1)
            if args.truth is not None:
                score_final(out / "final_forecast.csv", args.truth, out)
        elif args.command == "score":
            out = output_path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            for name in ("final_scored_rows.csv", "final_wape.json"):
                if (out / name).exists():
                    raise ValueError(f"Score output already exists: {out / name}")
            score_final(args.forecast, args.truth, out)
        else:
            from .smoke import check
            if args.full and args.out is None:
                raise ValueError("--full requires --out.")
            if args.full:
                require_models(include_arima=False)
            check(output_path(args.out) if args.full else None)
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(2, f"heat_ml: {error}\n")


if __name__ == "__main__":
    cli()

