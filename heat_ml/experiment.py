"""Run backtests, select on matched outer rows, and freeze the final forecast."""

import json

import numpy as np
import pandas as pd

from .baseline import KEYS, baselines
from .data import CONDITIONS, FINAL, HORIZON, ORIGINS, WEEK, load_target
from .features import Panel
from .models import ABLATIONS, GRIDS, forecast, wape

CLEAN_ORIGINS = (ORIGINS[1], ORIGINS[2])
HISTORICAL = {"arima_blend": {"mean4": 11.68, "mean2": 10.69},
              "arima_fourier_blend": {"mean4": 11.76, "mean2": 9.96}}


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_inputs(target, panel, conditions, membership):
    if panel.raw.loc[:FINAL].isna().all(axis=1).any() or panel.seoul.loc[:FINAL].isna().any():
        raise ValueError("Training data must contain observations and a Seoul total for every Sunday through 2025-12-28.")
    if not panel.units.size:
        raise ValueError("Empty mobility panel.")
    for name in set().union(*(CONDITIONS[c] for c in conditions)):
        if name not in panel.covariates:
            raise ValueError(f"Condition requires {name} data and release dates.")
        for origin in (*ORIGINS, FINAL):
            frame = panel.examples([origin])
            if frame[name].isna().all():
                raise ValueError(f"No available {name} observations for origin {origin.date()}.")
    if membership is not None and not set.union(*membership.values()) <= set(target.unit):
        raise ValueError("Historical membership contains unknown units.")


def run(target, covariates, membership, out, conditions, jobs, quick=False, historical=None, phase1=False):
    panel = Panel(target, covariates)
    check_inputs(target, panel, ("base",) if phase1 else conditions, membership)
    row_keys = ["origin", "unit", "week_end"]
    if historical is not None:
        historical = historical[row_keys].copy()
        historical["origin"] = pd.to_datetime(historical.origin)
        historical["week_end"] = pd.to_datetime(historical.week_end)
        historical["unit"] = historical.unit.astype(str)
        if historical.duplicated().any() or set(historical.origin) != set(ORIGINS):
            raise ValueError("Historical keys must be unique and include exactly the four outer origins.")
    keys = {(f"{model}|{condition}|{mode}"): (model, condition, mode)
            for model in GRIDS for condition in conditions for mode in ("standalone", "blend")}
    all_predictions, coverage, scores, dong_scores, tuning = [], [], [], [], {}
    run_kind = ("phase1_partial" if phase1 and membership is None else "phase1" if phase1 else
                "synthetic_smoke" if quick else
                "partial" if membership is None or tuple(conditions) != tuple(CONDITIONS) else "full")
    for origin in ORIGINS:
        print(f"Backtest {origin.date()}", flush=True)
        frame = panel.examples([origin])
        members = set(panel.units) if membership is None else membership[origin]
        baseline = baselines(panel, origin, frame, members, jobs, quick)
        wide = pd.concat([frame[["origin", "unit", "week_end", "h", "y"]], baseline], axis=1)
        eligible = wide.unit.isin(members)
        valid = eligible & np.isfinite(wide.y)
        tuning[str(origin.date())] = {"correction_factor": panel.factor(origin), "models": {}}
        for condition in conditions:
            for model in GRIDS:
                if phase1:
                    print(f"  {model} {condition}", flush=True)
                    pred, audit = forecast(panel, origin, model, condition, jobs, staged=True)
                else:
                    pred, audit = forecast(panel, origin, model, condition, jobs, quick)
                wide[f"{model}|{condition}|standalone"] = pred
                wide[f"{model}|{condition}|blend"] = (pred + wide.seasonal) / 2
                tuning[str(origin.date())]["models"][f"{model}|{condition}"] = audit

        columns = [*KEYS, *keys]
        in_history = pd.Series(True, index=wide.index)
        if historical is not None:
            origin_keys = pd.MultiIndex.from_frame(historical.loc[historical.origin.eq(origin)])
            in_history = pd.Series(pd.MultiIndex.from_frame(wide[row_keys]).isin(origin_keys), index=wide.index)
            if in_history.sum() != len(origin_keys) or (in_history & ~eligible).any():
                raise ValueError(f"Historical keys do not match the target grid or membership for {origin.date()}.")
        matched = valid & in_history & np.isfinite(wide[columns]).all(axis=1)
        wide["eligible_unit"] = eligible
        wide["in_historical_mask"] = in_history
        wide["matched"] = matched
        for key in columns:
            _, condition, _ = keys.get(key, (key, "base", "baseline"))
            coverage.append({
                "origin": origin, "candidate": key, "eligible_units": len(members),
                "eligible_rows": int(eligible.sum()), "missing_truth": int((eligible & ~valid).sum()),
                "missing_prediction_with_truth": int((valid & ~np.isfinite(wide[key])).sum()),
                "matched_rows": int(matched.sum()), "matched_units": wide.loc[matched, "unit"].nunique(),
                "historical_rows_missing_from_match": int((in_history & ~matched).sum())
                if historical is not None else None,
                **{f"{name}_missing_input_rows": int((eligible & frame[name].isna()).sum())
                   for name in CONDITIONS.get(condition, ())},
            })
        if not matched.any() or wide.loc[matched, "y"].sum() <= 0:
            raise ValueError(f"No positive matched comparison truth for {origin.date()}.")
        common = wide.loc[matched]
        denominator = common.groupby("unit").y.sum()
        error = common[columns].sub(common.y, axis=0).abs()
        dong = 100 * error.groupby(common.unit).sum().div(denominator.where(denominator.gt(0)), axis=0)
        dong["matched_rows"] = common.groupby("unit").size()
        dong["origin"] = origin
        for key in columns:
            scores.append({"origin": origin, "candidate": key, "wape": wape(common.y, common[key])})
        for key in keys:
            for baseline_key in KEYS:
                dong[f"{key} minus {baseline_key}"] = dong[key] - dong[baseline_key]
        dong_scores.append(dong.reset_index())
        all_predictions.append(wide)

    by_origin = pd.DataFrame(scores)
    average = by_origin.groupby("candidate").wape.mean().rename("mean4").to_frame()
    average["mean2"] = by_origin.loc[by_origin.origin.isin(CLEAN_ORIGINS)].groupby("candidate").wape.mean()
    average = average.sort_values(["mean4", "mean2"], kind="stable")
    if phase1:
        csv(pd.concat(all_predictions).assign(run_kind=run_kind), out / "outer_predictions.csv")
        csv(pd.DataFrame(coverage).assign(run_kind=run_kind), out / "coverage.csv")
        csv(pd.concat(dong_scores).assign(run_kind=run_kind), out / "dong_wape.csv")
        csv(by_origin.assign(run_kind=run_kind), out / "outer_wape.csv")
        csv(average.reset_index().assign(run_kind=run_kind), out / "mean_wape.csv")
        save_json(tuning, out / "outer_tuning.json")
        check_top_lightgbm(panel, all_predictions, average, out, jobs)
        return
    winner = average.loc[average.index.isin(keys)].index[0]
    model, condition, mode = keys[winner]

    print(f"Final refit 2025-12-28: {winner}", flush=True)
    prediction, audit = forecast(panel, FINAL, model, condition, jobs, quick)
    frame = panel.examples([FINAL])
    final = frame[["origin", "unit", "week_end", "h"]].copy()
    final["week_start"] = final.week_end - pd.Timedelta(days=6)
    final["y_hat"] = prediction if mode == "standalone" else (prediction + frame.seasonal) / 2
    final["candidate"] = winner
    final["run_kind"] = run_kind

    csv(pd.concat(all_predictions).assign(run_kind=run_kind), out / "outer_predictions.csv")
    csv(pd.DataFrame(coverage).assign(run_kind=run_kind), out / "coverage.csv")
    csv(pd.concat(dong_scores).assign(run_kind=run_kind), out / "dong_wape.csv")
    csv(by_origin.assign(run_kind=run_kind), out / "outer_wape.csv")
    csv(average.reset_index().assign(run_kind=run_kind), out / "mean_wape.csv")
    csv(final, out / "final_forecast.csv")
    save_json(tuning, out / "outer_tuning.json")
    save_json({
        "selected": winner, "selection": "minimum outer mean4; mean2 tiebreak",
        "mode": run_kind,
        "historical_416_membership_supplied": membership is not None,
        "comparison": "Intersection of finite raw truth and all candidate and rerun baseline predictions.",
        "historical_rows": (
            "Historical mask supplied. See coverage.csv for rows missing from the match."
            if historical is not None else
            "Historical t7 used additional models in its row mask; equivalence to that mask is unverified."
        ),
        "retrospective_reference_only": HISTORICAL,
        "final_cutoff": str(FINAL.date()), "final_correction_factor": panel.factor(FINAL),
        "final_tuning": audit,
    }, out / "selection.json")


def check_top_lightgbm(panel, predictions, average, out, jobs):
    top = [key for key in average.index if key.startswith("lightgbm|") and key.endswith("|standalone")][:2]
    scores, tuning, horizons = [], {}, []
    for wide, origin in zip(predictions, ORIGINS):
        for candidate in top:
            _, condition, _ = candidate.split("|")
            print(f"Full-grid check {origin.date()} {condition}", flush=True)
            pred, audit = forecast(panel, origin, "lightgbm", condition, jobs)
            tuning[f"{origin.date()}|{condition}"] = audit
            wide[f"full_grid|{condition}|standalone"] = pred
            wide[f"full_grid|{condition}|blend"] = (pred + wide.seasonal) / 2
            matched = wide.loc[wide.matched]
            for mode in ("standalone", "blend"):
                scores.append({"origin": origin, "condition": condition, "mode": mode,
                               "staged_wape": wape(matched.y, matched[f"lightgbm|{condition}|{mode}"]),
                               "full_grid_wape": wape(matched.y, matched[f"full_grid|{condition}|{mode}"])})
        matched = wide.loc[wide.matched]
        columns = [*KEYS, *(f"{model}|{condition}|{mode}" for model in GRIDS
                            for condition in ABLATIONS for mode in ("standalone", "blend")),
                   *(column for column in wide if column.startswith("full_grid|"))]
        for low, high in ((1, 4), (5, 8), (9, 13)):
            block = matched.loc[matched.h.between(low, high)]
            horizons.extend({"origin": origin, "horizon": f"{low}-{high}", "candidate": column,
                             "wape": wape(block.y, block[column])} for column in columns)
    csv(pd.DataFrame(scores), out / "lightgbm_full_grid_check.csv")
    csv(pd.DataFrame(horizons), out / "horizon_wape.csv")
    save_json(tuning, out / "lightgbm_full_grid_tuning.json")


def score_final(forecast_path, truth_path, out):
    frozen = pd.read_csv(forecast_path, dtype={"unit": str}, parse_dates=["week_end", "origin"])
    if "run_kind" in frozen and frozen.run_kind.eq("synthetic_smoke").any():
        raise ValueError("Synthetic smoke runs cannot be scored as 2026 forecasts.")
    expected = set(pd.date_range(FINAL + WEEK, periods=HORIZON, freq="W-SUN"))
    if (not frozen.origin.eq(FINAL).all() or set(frozen.week_end) != expected
            or frozen.duplicated(["unit", "week_end"]).any()
            or not frozen.groupby("unit").week_end.size().eq(HORIZON).all()):
        raise ValueError("Expected a fixed 13-week forecast made on 2025-12-28.")
    truth = load_target(truth_path)
    result = frozen.merge(truth[["unit", "week_end", "y"]], on=["unit", "week_end"],
                          how="left", validate="one_to_one")
    matched = np.isfinite(result.y) & np.isfinite(result.y_hat)
    valid = result.loc[matched]
    if valid.empty or valid.y.sum() <= 0:
        raise ValueError("No usable 2026 truth on frozen forecast keys.")
    result["scored"] = matched
    csv(result, out / "final_scored_rows.csv")
    save_json({
        "wape": wape(valid.y, valid.y_hat), "matched_rows": len(valid),
        "forecast_rows": len(result), "matched_weeks": valid.week_end.nunique(),
        "matched_units": valid.unit.nunique(), "missing_truth": int(result.y.isna().sum()),
        "missing_forecasts": int(result.y_hat.isna().sum()),
    }, out / "final_wape.json")

