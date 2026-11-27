"""v3 experiments: one LightGBM model, simple anomaly handling, simple features, no blends.

Tuning and validation are unchanged from v2 (staged LightGBM grid, two nested 13-week folds,
four outer origins, raw truth, clean and segment WAPE). Each condition is scored as the model's
standalone prediction only.

Step 0:  python -m heat_ml.v3 --correction none --conditions base_lag0 --out heat_ml/run_v3_s0_raw
         python -m heat_ml.v3 --correction mask --conditions base_lag0 --out heat_ml/run_v3_s0_mask
Step 1:  python -m heat_ml.v3 --correction daily_jan_did --festival --semester \
             --conditions minimal minimal+lag0 minimal+lag0+recent_lags ... --out heat_ml/run_v3_s1
"""

import argparse
from pathlib import Path

import pandas as pd

from .correction import VARIANTS, make_panel
from .data import ORIGINS, load_target
from .experiment import CLEAN_ORIGINS, csv, save_json
from .festival import FestivalMixin, with_mixins
from .kr_calendar import CONFIRMED, WINDOWS
from .models import forecast
from .segments import FESTIVAL_MIN, SEMESTER_MIN, STEP_MIN, dong_types, week_types
from .v2 import KEYS, score

PACKAGE = Path(__file__).resolve().parent
MODEL = "lightgbm"
V3_CORRECTIONS = tuple(VARIANTS)  # v3 runs use daily_jan_did (v2 correction); none/mask failed in Step 0


def predict_origin(panel, origin, conditions, jobs, path):
    if path.is_file():
        print(f"Reusing {path.name}", flush=True)
        return pd.read_csv(path, dtype={"unit": str}, parse_dates=["origin", "week_end"]), None
    print(f"Backtest {origin.date()}", flush=True)
    frame = panel.examples([origin])
    wide = frame[[*KEYS, "h", "y"]].copy()
    audits = {}
    for condition in conditions:
        print(f"  {MODEL} {condition}", flush=True)
        prediction, audit = forecast(panel, origin, MODEL, condition, jobs, staged=True)
        wide[f"{MODEL}|{condition}"] = prediction
        audits[f"{MODEL}|{condition}"] = audit
    temp = path.with_suffix(".tmp")
    csv(wide, temp)
    temp.replace(path)
    return wide, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correction", choices=V3_CORRECTIONS, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--target", type=Path, default=PACKAGE / "target_2023_2025.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--festival", action="store_true", help="Enable holiday/festival calendar columns")
    parser.add_argument("--semester", action="store_true", help="Enable university calendar columns")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=True)

    from .semester import SemesterMixin
    target = load_target(args.target)
    panel = make_panel(args.correction, target)
    panel = with_mixins(panel, *([FestivalMixin] if args.festival else []),
                        *([SemesterMixin] if args.semester else []))
    dongs = dong_types(target)
    weeks = week_types(panel.weeks)
    csv(dongs, out / "dong_types.csv")
    save_json({
        "version": "v3", "model": MODEL, "output": "standalone only (no blend, no alignment)",
        "correction": args.correction, "windows": None if VARIANTS[args.correction] is None else list(VARIANTS[args.correction]),
        "window_dates": {name: [str(day.date()) for day in WINDOWS[name]] for name in WINDOWS},
        "clean_scope_excludes_weeks_with_days_in": CONFIRMED,
        "units": int(len(panel.units)), "conditions": args.conditions,
        "festival_features": args.festival, "semester_features": args.semester,
        "complete_window_means": True, "lightgbm_tuning": "staged",
        "validation": "two nested 13-week folds before each origin (same as v2)",
        "truth": "raw weekly daily-mean mobility (uncorrected)",
        "mean2_origins": [str(origin.date()) for origin in CLEAN_ORIGINS],
        "dong_type_rules": {"step": STEP_MIN, "university": SEMESTER_MIN, "festival_residential": FESTIVAL_MIN},
        "dong_type_counts": dongs.dong_type.value_counts().to_dict(),
    }, out / "setup.json")

    predictions = []
    for origin in ORIGINS:
        wide, audits = predict_origin(panel, origin, args.conditions, args.jobs,
                                      out / f"predictions_{origin.date()}.csv")
        predictions.append(wide)
        if audits is not None:
            save_json({"correction": panel.factor(origin), "models": audits}, out / f"tuning_{origin.date()}.json")

    predictions = pd.concat(predictions, ignore_index=True)
    columns = [column for column in predictions.columns if column.startswith(f"{MODEL}|")]
    mean, by_origin, segments, coverage, data = score(predictions, columns, dongs, weeks)
    csv(mean, out / "mean_wape.csv")
    csv(by_origin, out / "outer_wape.csv")
    csv(segments, out / "segment_wape.csv")
    csv(coverage, out / "coverage.csv")
    csv(data, out / "outer_predictions.csv")
    print(mean.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
