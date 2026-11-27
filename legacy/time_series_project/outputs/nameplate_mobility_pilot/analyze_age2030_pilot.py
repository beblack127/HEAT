"""Paired diagnostics for the age 20s+30s 4+5+7 exploratory forecast."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_pilot"
pred = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
age = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                 for p in sorted((ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_mobility").glob("age2030_457_20????.csv"))], ignore_index=True)
monthly = pred.groupby(["model", "month"], as_index=False).agg(abs_error=("abs_error", "sum"), actual=("actual_daily", "sum"))
monthly["wape_pct"] = 100 * monthly.abs_error / monthly.actual
monthly.to_csv(OUT / "monthly_wape.csv", index=False, encoding="utf-8-sig")
dong = pred.groupby(["model", "admin_dong_code"], as_index=False).agg(abs_error=("abs_error", "sum"), actual=("actual_daily", "sum"))
dong["wape_pct"] = 100 * dong.abs_error / dong.actual
dong.to_csv(OUT / "per_dong_wape.csv", index=False, encoding="utf-8-sig")

base = dong.loc[dong.model.eq("SARIMA")].set_index("admin_dong_code")
metro = dong.loc[dong.model.eq("SARIMAX_subway")].set_index("admin_dong_code")
paired = base[["abs_error", "wape_pct"]].join(metro[["abs_error", "wape_pct"]], suffixes=("_base", "_subway"))
paired["abs_error_reduction"] = paired.abs_error_base - paired.abs_error_subway
paired.to_csv(OUT / "subway_paired_dong.csv", encoding="utf-8-sig")

rng = np.random.default_rng(20261115)
values = paired[["abs_error_base", "abs_error_subway"]].to_numpy()
actual = base.loc[paired.index, "actual"].to_numpy()
draws = []
for _ in range(3000):
    idx = rng.integers(0, len(values), len(values))
    denom = actual[idx].sum()
    draws.append(100 * (values[idx, 0].sum() - values[idx, 1].sum()) / denom)

prior = age[["admin_dong_code", "month", "mobility_457_age20_39_daily"]].copy()
prior["month"] = (prior.month.str[:4].astype(int) + 1).astype(str) + prior.month.str[4:]
prior = prior.rename(columns={"mobility_457_age20_39_daily": "prior_year_daily"})
rank = pred.merge(prior, on=["admin_dong_code", "month"], how="left", validate="many_to_one")
rank = rank.loc[rank.month.le("2025-07")].copy()  # 2024-08 has a broad source anomaly.
rank["actual_growth"] = np.log1p(rank.actual_daily) - np.log1p(rank.prior_year_daily)
rank["predicted_growth"] = np.log1p(rank.predicted_daily) - np.log1p(rank.prior_year_daily)
top10 = {}
for name, model in rank.groupby("model"):
    hits = []
    for month, frame in model.groupby("month"):
        actual_top = set(frame.nlargest(10, "actual_growth").admin_dong_code)
        pred_top = set(frame.nlargest(10, "predicted_growth").admin_dong_code)
        hits.append({"month": month, "hits": len(actual_top & pred_top)})
    top10[name] = {"hits": sum(x["hits"] for x in hits), "out_of": 70, "by_month": hits}

report = {
    "subway_vs_sarima": {
        "dongs_with_lower_error": int(paired.abs_error_reduction.gt(0).sum()),
        "dongs_with_higher_error": int(paired.abs_error_reduction.lt(0).sum()),
        "wape_reduction_pct_points": round(100 * paired.abs_error_reduction.sum() / actual.sum(), 3),
        "dong_bootstrap_95pct_points": [round(float(x), 3) for x in np.quantile(draws, [.025, .975])],
        "largest_single_dong_fraction_of_gain": round(float(paired.abs_error_reduction.max() / paired.abs_error_reduction.sum()), 3),
    },
    "top10_growth_40d_2025jan_jul": top10,
    "notes": "Top-10 is exploratory and computed after seeing WAPE. Dong bootstrap treats dong as cluster but does not resolve shared city shocks or selected pilot bias."
}
(OUT / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
