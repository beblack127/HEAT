"""Exploratory top-K growth ranking on the existing 40-dong predictions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRED = pd.read_csv(OUT / "predictions.csv", encoding="utf-8-sig", dtype={"admin_dong_code": str})
folder = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly"
prev = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                  for p in sorted(folder.glob("mobility_2024??.csv"))], ignore_index=True)
prev["month"] = (pd.PeriodIndex(prev.month, freq="M") + 12).astype(str)
prev = prev.rename(columns={"mobility_shop_tour_other_daily": "last_year_daily"})
df = PRED.merge(prev[["admin_dong_code", "month", "last_year_daily"]],
                on=["admin_dong_code", "month"], how="left", validate="many_to_one")
assert len(df) == len(PRED) and df.last_year_daily.gt(0).all()
df["actual_growth"] = df.actual_daily / df.last_year_daily - 1
df["forecast_growth"] = df.predicted_daily / df.last_year_daily - 1
result = {}
for k in (5, 10):
    result[k] = {}
    for model, g in df.groupby("model"):
        month_scores = []
        for month, h in g.groupby("month"):
            actual_top = set(h.nlargest(k, "actual_growth").admin_dong_code)
            predicted_top = set(h.nlargest(k, "forecast_growth").admin_dong_code)
            month_scores.append({"month": month, "hits": len(actual_top & predicted_top),
                                 "precision": len(actual_top & predicted_top) / k})
        result[k][model] = {"mean_precision": round(sum(x["precision"] for x in month_scores)/len(month_scores), 3),
                            "hits_out_of": f"{sum(x['hits'] for x in month_scores)}/{k * len(month_scores)}",
                            "by_month": month_scores}
(OUT / "rank_diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {m: {"mean_precision": r["mean_precision"], "hits_out_of": r["hits_out_of"]}
                      for m, r in models.items()} for k, models in result.items()}, indent=2))
