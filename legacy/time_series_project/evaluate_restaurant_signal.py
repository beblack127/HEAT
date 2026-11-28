"""Point-in-time restaurant ablation for the existing named-place pilot.

The target quarter is forecast with trade-area data through t-2. Restaurants
are candidate predictors, never added to the Hotness Score target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tmp/pythonlibs"))
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = ROOT / "results/semester_pilot"
PANEL = OUT / "trade_food_quarterly_2023_2025.csv"
BRIDGE = OUT / "place_trade_bridge_provisional.csv"


def metrics(df: pd.DataFrame) -> dict:
    a = df.actual.to_numpy(float)
    p = df.forecast.to_numpy(float)
    return {
        "n": int(len(df)),
        "wape": round(float(np.abs(a - p).sum() / np.abs(a).sum()), 4),
        "mae": round(float(np.abs(a - p).mean()), 2),
        "bias": round(float((p - a).mean()), 2),
    }


def main() -> None:
    data = pd.read_csv(PANEL, dtype={"trade_code": str, "quarter": str})
    data = data.sort_values(["trade_code", "quarter"])
    data = data.groupby("trade_code").filter(
        lambda x: len(x) == 12 and x.estimated_footfall.notna().all()
    ).copy()
    group = data.groupby("trade_code")
    for lag in (2, 3, 4):
        data[f"log_footfall_lag{lag}"] = np.log1p(group.estimated_footfall.shift(lag))
    for col in ("store_count", "openings", "closures"):
        data[f"{col}_lag2"] = group[col].shift(2)
    data["net_openings_lag2"] = data.openings_lag2 - data.closures_lag2
    data["quarter_index"] = data.quarter.str[-1].astype(int)
    common = [f"log_footfall_lag{lag}" for lag in (2, 3, 4)] + ["quarter_index"]
    variants = {
        "lag_only": common,
        "plus_store_stock": common + ["store_count_lag2"],
        "plus_net_openings": common + ["net_openings_lag2"],
        "plus_openings_closures": common + ["openings_lag2", "closures_lag2"],
        "plus_all_restaurant": common + ["store_count_lag2", "openings_lag2", "closures_lag2"],
    }
    records = []
    quarters = sorted(data.quarter.unique())
    for target in ("20251", "20252", "20253", "20254"):
        last_published = quarters[quarters.index(target) - 2]
        train = data[(data.quarter <= last_published) & data[common].notna().all(axis=1)]
        hold = data[(data.quarter == target) & data[common].notna().all(axis=1)]
        assert len(train) > 1000 and len(hold) > 1000
        variants_for_origin = {"last_published": np.expm1(hold.log_footfall_lag2.to_numpy())}
        for name, features in variants.items():
            assert train[features].notna().all().all() and hold[features].notna().all().all()
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
            model.fit(train[features], np.log1p(train.estimated_footfall))
            variants_for_origin[name] = np.expm1(model.predict(hold[features])).clip(0)
        for name, forecast in variants_for_origin.items():
            for code, actual, pred in zip(
                hold.trade_code, hold.estimated_footfall, forecast
            ):
                records.append(
                    {"trade_code": code, "target_quarter": target,
                     "model": name, "actual": actual, "forecast": float(pred)}
                )
    pred = pd.DataFrame(records)
    pred.to_csv(OUT / "restaurant_footfall_ablation_2025.csv", index=False, encoding="utf-8-sig")

    bridge = pd.read_csv(BRIDGE, dtype={"trade_code": str})
    score = pd.read_csv(OUT / "named_place_hotness_score_quarterly.csv", dtype={"quarter": str})
    eligible = score[(score.quarter.str.startswith("2025")) & score.score_valid]
    areas = set(eligible.search_area)
    matched = pred.merge(bridge[["trade_code", "search_area"]], on="trade_code", validate="many_to_one")
    matched = matched[matched.search_area.isin(areas)]
    # Place totals are comparable across variants on identical trade-area rows.
    places = matched.groupby(["search_area", "target_quarter", "model"], as_index=False)[["actual", "forecast"]].sum()
    places.to_csv(OUT / "restaurant_place_footfall_ablation_2025.csv", index=False, encoding="utf-8-sig")
    search = pd.read_csv(OUT / "search_3month_backtest_predictions.csv")
    search = search[search.model == "same_month_last_year"].copy()
    search["quarter"] = search.target_month.str[:4] + (((search.target_month.str[5:7].astype(int) - 1) // 3) + 1).astype(str)
    search = search.groupby(["search_area", "quarter"], as_index=False).agg(pred_search=("forecast", "mean"))
    reference = score[(score.quarter >= "20241") & (score.quarter <= "20244")]
    cols = ["search_interest_mean", "search_interest_mean_yoy", "footfall_per_trade", "footfall_per_trade_yoy"]
    refs = {col: np.sort(reference[col].dropna().to_numpy()) for col in cols}
    prior = score[["search_area", "quarter", "search_interest_mean", "footfall_per_trade"]].copy()
    prior["quarter"] = (prior.quarter.str[:4].astype(int) + 1).astype(str) + prior.quarter.str[-1]
    prior = prior.rename(columns={"search_interest_mean": "prior_search", "footfall_per_trade": "prior_foot"})
    joined = places.rename(columns={"target_quarter": "quarter"}).merge(
        eligible[["search_area", "quarter", "trade_count", "hotness_score"]],
        on=["search_area", "quarter"], validate="many_to_one"
    ).merge(search, on=["search_area", "quarter"], validate="many_to_one").merge(
        prior, on=["search_area", "quarter"], validate="many_to_one"
    )
    assert len(joined) == 40 * (len(variants) + 1), len(joined)
    joined["pred_foot_per_trade"] = joined.forecast / joined.trade_count
    joined["pred_search_yoy"] = (joined.pred_search / joined.prior_search - 1).clip(-1, 3)
    joined["pred_foot_yoy"] = (joined.pred_foot_per_trade / joined.prior_foot - 1).clip(-1, 3)
    def pct(col: str, values: pd.Series) -> np.ndarray:
        return np.searchsorted(refs[col], values.to_numpy(), side="right") / len(refs[col]) * 100
    joined["pred_score"] = (
        .35 * pct("search_interest_mean", joined.pred_search)
        + .15 * pct("search_interest_mean_yoy", joined.pred_search_yoy)
        + .35 * pct("footfall_per_trade", joined.pred_foot_per_trade)
        + .15 * pct("footfall_per_trade_yoy", joined.pred_foot_yoy)
    )
    joined.to_csv(OUT / "restaurant_score_ablation_2025.csv", index=False, encoding="utf-8-sig")
    results = {
        "availability": "Footfall and restaurant features through target quarter minus 2; 2025 held out; no target-quarter restaurant values used.",
        "all_trade_footfall": {name: metrics(g) for name, g in pred.groupby("model")},
        "matched_place_footfall": {name: metrics(g) for name, g in places.groupby("model")},
        "score": {},
        "eligible_places": sorted(areas),
    }
    for name, g in joined.groupby("model"):
        quarter = []
        for q, z in g.groupby("quarter"):
            actual_top = set(z.nlargest(3, "hotness_score").search_area)
            forecast_top = set(z.nlargest(3, "pred_score").search_area)
            quarter.append({"quarter": q, "spearman": float(z.hotness_score.rank().corr(z.pred_score.rank())),
                            "precision_at_3": len(actual_top & forecast_top) / 3})
        results["score"][name] = {
            "n": len(g), "mae": round(float((g.hotness_score - g.pred_score).abs().mean()), 3),
            "mean_spearman": round(float(np.mean([x["spearman"] for x in quarter])), 3),
            "precision_at_3": round(float(np.mean([x["precision_at_3"] for x in quarter])), 3),
            "quarters": quarter,
        }
    (OUT / "restaurant_ablation_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
