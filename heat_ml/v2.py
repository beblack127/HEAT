"""v2 feature experiments: new anomaly correction, 426 dongs, clean and segment WAPE.

python -m heat_ml.v2 --correction daily --out heat_ml/run_v2_baseline_daily --jobs 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import baselines
from .correction import VARIANTS, make_panel
from .data import ORIGINS, load_target
from .experiment import CLEAN_ORIGINS, csv, save_json
from .festival import FestivalMixin, with_mixins
from .level import LevelMixin
from .robust import SHARE_OFFSET, RobustMixin
from .semester import SemesterMixin
from .kr_calendar import CONFIRMED, WINDOWS
from .models import forecast, wape
from .segments import FESTIVAL_MIN, SEMESTER_MIN, STEP_MIN, dong_types, week_types

PACKAGE = Path(__file__).resolve().parent
KEYS = ["origin", "unit", "week_end"]
BASELINE = ("base_lag0",)
REFERENCES = (("lightgbm", "only_means"), ("ridge", "only_fourier"))
HORIZONS = ((1, 13), (1, 4), (5, 8), (9, 13))
SCOPES = ("all", "clean")
# Alternative prior-year values; each present column adds a "blend_<suffix>" mode.
BLEND_REFERENCES = ("seasonal_aligned", "seasonal_level", "seasonal_aligned_level")


def candidates(conditions, references=True):
    pairs = [(model, condition) for condition in conditions for model in ("ridge", "lightgbm")]
    return pairs + [pair for pair in REFERENCES if references and pair not in pairs]


def predict_origin(panel, origin, pairs, jobs, arima, path):
    if path.is_file():
        print(f"Reusing {path.name}", flush=True)
        return pd.read_csv(path, dtype={"unit": str}, parse_dates=["origin", "week_end"]), None
    print(f"Backtest {origin.date()}", flush=True)
    frame = panel.examples([origin])
    references = [column for column in BLEND_REFERENCES if column in frame]
    wide = frame[[*KEYS, "h", "y", "seasonal", *references]].copy()
    if arima:
        extra = baselines(panel, origin, frame, set(panel.units), jobs)
        wide[["arima_blend", "arima_fourier_blend"]] = extra[["arima_blend", "arima_fourier_blend"]].to_numpy()
    audits = {}
    for model, condition in pairs:
        print(f"  {model} {condition}", flush=True)
        base, share = condition.removesuffix("@share"), condition.endswith("@share")
        options = dict(offset_column=SHARE_OFFSET, offset_space="log") if share else {}
        prediction, audit = forecast(panel, origin, model, base, jobs, staged=True, **options)
        wide[f"{model}|{condition}|standalone"] = prediction
        wide[f"{model}|{condition}|blend"] = (prediction + wide.seasonal.to_numpy()) / 2
        for column in references:
            suffix = column.removeprefix("seasonal_")
            wide[f"{model}|{condition}|blend_{suffix}"] = (prediction + wide[column].to_numpy()) / 2
        audits[f"{model}|{condition}"] = audit
    temp = path.with_suffix(".tmp")
    csv(wide, temp)
    temp.replace(path)
    return wide, audits


def score(predictions, columns, dongs, weeks):
    data = predictions.merge(dongs[["unit", "dong_type"]], on="unit", how="left", validate="many_to_one")
    data = data.merge(weeks[["week_end", "week_type", "confirmed_anomaly_days", "jan2025_days"]],
                      on="week_end", how="left", validate="many_to_one")
    data["matched"] = np.isfinite(data.y) & np.isfinite(data[columns]).all(axis=1)
    data["clean"] = data.confirmed_anomaly_days.eq(0)
    rows, segments = [], []
    for scope in SCOPES:
        scoped = data.loc[data.matched & (data.clean if scope == "clean" else True)]
        for origin, block in scoped.groupby("origin"):
            for low, high in HORIZONS:
                part = block.loc[block.h.between(low, high)]
                rows.extend({"scope": scope, "origin": origin, "horizon": f"{low}-{high}", "candidate": column,
                             "rows": len(part), "weeks": part.week_end.nunique(),
                             "wape": wape(part.y, part[column]) if len(part) else np.nan} for column in columns)
        for kind in ("dong_type", "week_type"):
            for (segment, origin), block in [*scoped.groupby([kind, "origin"]),
                                             *(((name, "pooled"), part) for name, part in scoped.groupby(kind))]:
                segments.extend({"scope": scope, "segment_kind": kind, "segment": segment,
                                 "origin": origin if origin == "pooled" else str(origin.date()),
                                 "candidate": column, "rows": len(block), "units": block.unit.nunique(),
                                 "weeks": block.week_end.nunique(), "wape": wape(block.y, block[column])}
                                for column in columns)
    by_origin = pd.DataFrame(rows)
    full = by_origin.loc[by_origin.horizon.eq("1-13")]
    clean_origins = full.origin.isin(CLEAN_ORIGINS)
    mean = pd.DataFrame({
        "mean4": full.loc[full.scope.eq("all")].groupby("candidate").wape.mean(),
        "mean2": full.loc[full.scope.eq("all") & clean_origins].groupby("candidate").wape.mean(),
        "mean4_clean": full.loc[full.scope.eq("clean")].groupby("candidate").wape.mean(),
        "mean2_clean": full.loc[full.scope.eq("clean") & clean_origins].groupby("candidate").wape.mean(),
    }).sort_values("mean4").rename_axis("candidate").reset_index()
    flags = data.assign(clean_matched=data.clean & data.matched,
                        clean_week=data.week_end.where(data.clean),
                        jan2025_week=data.week_end.where(data.jan2025_days.gt(0)))
    coverage = flags.groupby("origin").agg(rows=("y", "size"), matched=("matched", "sum"),
                                           clean_matched=("clean_matched", "sum"),
                                           clean_weeks=("clean_week", "nunique"),
                                           jan2025_weeks=("jan2025_week", "nunique"))
    return mean, by_origin, pd.DataFrame(segments), coverage.reset_index(), data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--conditions", nargs="+", default=list(BASELINE))
    parser.add_argument("--target", type=Path, default=PACKAGE / "target_2023_2025.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-arima", action="store_true")
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument("--festival", action="store_true", help="Add festival columns and aligned blends")
    parser.add_argument("--semester", action="store_true", help="Add university calendar columns")
    parser.add_argument("--level", action="store_true", help="Add level-ratio and step-change columns")
    parser.add_argument("--robust", action="store_true",
                        help="Add robust lags, Seoul/share components, anomaly flags; enables '<condition>@share'")
    parser.add_argument("--drop-pre-shift", action="store_true",
                        help="P7: drop training rows before a dong's detected step change (implies --level)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PACKAGE) or out == PACKAGE:
        parser.error("Output must be inside heat_ml/.")
    out.mkdir(parents=True, exist_ok=True)

    target = load_target(args.target)
    panel = make_panel(args.correction, target)
    level = args.level or args.drop_pre_shift
    panel = with_mixins(panel, *([FestivalMixin] if args.festival else []),
                        *([SemesterMixin] if args.semester else []), *([LevelMixin] if level else []),
                        *([RobustMixin] if args.robust else []))
    if any(condition.endswith("@share") for condition in args.conditions) and not args.robust:
        parser.error("'@share' conditions need --robust.")
    panel.drop_pre_shift = args.drop_pre_shift
    pairs = candidates(args.conditions, not args.no_references)
    dongs = dong_types(target)
    weeks = week_types(panel.weeks)
    csv(dongs, out / "dong_types.csv")
    save_json({
        "correction": args.correction, "windows": VARIANTS[args.correction],
        "window_dates": {name: [str(day.date()) for day in WINDOWS[name]] for name in WINDOWS},
        "clean_scope_excludes_weeks_with_days_in": CONFIRMED,
        "units": int(len(panel.units)), "membership": "all 426 dongs (no historical 416 list)",
        "candidates": [f"{model}|{condition}" for model, condition in pairs], "arima": not args.no_arima,
        "festival_features": args.festival, "semester_features": args.semester,
        "level_features": level, "drop_pre_shift": args.drop_pre_shift, "robust_features": args.robust,
        "complete_window_means": True, "lightgbm_tuning": "staged",
        "truth": "raw weekly daily-mean mobility (uncorrected)",
        "mean2_origins": [str(origin.date()) for origin in CLEAN_ORIGINS],
        "dong_type_rules": {"step": STEP_MIN, "university": SEMESTER_MIN, "festival_residential": FESTIVAL_MIN},
        "dong_type_counts": dongs.dong_type.value_counts().to_dict(),
    }, out / "setup.json")

    predictions, tuning = [], {}
    for origin in ORIGINS:
        wide, audits = predict_origin(panel, origin, pairs, args.jobs, not args.no_arima,
                                      out / f"predictions_{origin.date()}.csv")
        predictions.append(wide)
        if audits is not None:
            tuning[str(origin.date())] = {"correction": panel.factor(origin), "models": audits}
            path = out / f"tuning_{origin.date()}.json"
            save_json(tuning[str(origin.date())], path)

    predictions = pd.concat(predictions, ignore_index=True)
    columns = [column for column in predictions.columns
               if "|" in column or column.startswith("arima") or column.startswith("seasonal")]
    mean, by_origin, segments, coverage, data = score(predictions, columns, dongs, weeks)
    csv(mean, out / "mean_wape.csv")
    csv(by_origin, out / "outer_wape.csv")
    csv(segments, out / "segment_wape.csv")
    csv(coverage, out / "coverage.csv")
    csv(data, out / "outer_predictions.csv")
    print(mean.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
