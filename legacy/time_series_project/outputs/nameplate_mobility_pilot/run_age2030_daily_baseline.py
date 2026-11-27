"""Build daily 20s+30s purpose-4/5/7 arrivals and leakage-safe naive baselines."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "nameplate_mobility_pilot"
OUT = BASE / "age2030_daily_baseline"
OUT.mkdir(parents=True, exist_ok=True)
MONTHLY = BASE / "age2030_mobility"
EVAL_START, EVAL_END = "2025-01-01", "2025-08-31"


def aggregate_month(month: str) -> dict:
    source = Path(tempfile.gettempdir()) / f"codex_seoul_mobility_{month}.zip"
    target = OUT / f"daily_457_age2030_{month}.csv"
    if target.exists():
        frame = pd.read_csv(target)
        return {"month": month, "days": int(frame.date.nunique()), "rows": len(frame), "cached": True}
    if not zipfile.is_zipfile(source):
        raise FileNotFoundError(source)
    sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            if not member.endswith(".csv"):
                continue
            with archive.open(member) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                header = next(reader)
                ix = {k: header.index(k) for k in (
                    "etl_ymd", "d_admdong_cd", "move_purpose", "male_20_cnt", "feml_20_cnt",
                    "male_30_cnt", "feml_30_cnt", "total_cnt")}
                for row in reader:
                    if row[ix["move_purpose"]] not in {"4", "5", "7"}:
                        continue
                    code = row[ix["d_admdong_cd"]]
                    if len(code) != 8 or not code.startswith("11"):
                        continue
                    date = row[ix["etl_ymd"]]
                    a = sums[(date, code)]
                    a[0] += float(row[ix["male_20_cnt"]]) + float(row[ix["feml_20_cnt"]])
                    a[1] += float(row[ix["male_30_cnt"]]) + float(row[ix["feml_30_cnt"]])
                    a[2] += float(row[ix["total_cnt"]])
    rows = [{"date": d, "admin_dong_code": c, "age20": v[0], "age30": v[1],
             "age20_39": v[0] + v[1], "all_ages_457": v[2]}
            for (d, c), v in sorted(sums.items())]
    dates = {r["date"] for r in rows}
    if len(dates) < 27:
        raise ValueError(f"Incomplete month {month}: {len(dates)} dates")
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Independent monthly aggregate from the preceding experiment.
    monthly = pd.read_csv(MONTHLY / f"age2030_457_{month}.csv", dtype={"admin_dong_code": str})
    check = pd.DataFrame(rows).groupby("admin_dong_code").age20_39.sum() / len(dates)
    reference = monthly.set_index("admin_dong_code").mobility_457_age20_39_daily
    if set(check.index) != set(reference.index) or not np.allclose(check.loc[reference.index], reference, atol=1e-5):
        raise ValueError(f"Daily/monthly age target mismatch: {month}")
    return {"month": month, "days": len(dates), "rows": len(rows), "cached": False,
            "max_monthly_difference": float((check.loc[reference.index] - reference).abs().max())}


def evaluate() -> dict:
    files = sorted(OUT.glob("daily_457_age2030_20????.csv"))
    if len(files) != 32:
        raise ValueError(f"Expected 32 daily month files, got {len(files)}")
    panel = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                       for p in files], ignore_index=True)
    panel["date"] = pd.to_datetime(panel.date.astype(str), format="%Y%m%d")
    if panel.duplicated(["date", "admin_dong_code"]).any():
        raise ValueError("Duplicate dong-date")
    panel["month"] = panel.date.dt.strftime("%Y-%m")
    monthly_check = panel.groupby(["admin_dong_code", "month"], as_index=False).age20_39.mean()
    monthly_ref = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                             for p in sorted(MONTHLY.glob("age2030_457_20????.csv"))], ignore_index=True)
    monthly_check = monthly_check.merge(monthly_ref[["admin_dong_code", "month", "mobility_457_age20_39_daily"]],
                                        on=["admin_dong_code", "month"], validate="one_to_one")
    if len(monthly_check) != len(monthly_ref):
        raise ValueError("Daily/monthly coverage mismatch")
    max_monthly_difference = float((monthly_check.age20_39 - monthly_check.mobility_457_age20_39_daily).abs().max())
    if max_monthly_difference > 1e-5:
        raise ValueError("Daily/monthly target mismatch")
    # Reindex each dong to actual calendar dates so shifts mean elapsed days.
    codes = sorted(panel.admin_dong_code.unique())
    dates = pd.date_range(panel.date.min(), panel.date.max(), freq="D")
    index = pd.MultiIndex.from_product([codes, dates], names=["admin_dong_code", "date"])
    y = panel.set_index(["admin_dong_code", "date"]).age20_39.reindex(index)
    if y.notna().mean() < .98:
        raise ValueError("Unexpected daily coverage gap")
    frame = y.rename("actual").to_frame()
    # For a target date D, only values through D-h are available. 91 days
    # represents the approximately three-month, same-weekday origin.
    specs = [(1, "last_observed", 1), (1, "same_weekday", 7),
             (7, "last_observed", 7),
             (30, "last_observed", 30), (30, "same_weekday", 35),
             (91, "last_observed", 91), (91, "same_weekday_year_ago", 364)]
    specs.extend((h, "trailing_7day_mean", h) for h in (1, 7, 30, 91))
    outputs = []
    all_dongs = frame.index.get_level_values("admin_dong_code")
    all_dates = frame.index.get_level_values("date")
    test = (all_dates >= pd.Timestamp(EVAL_START)) & (all_dates <= pd.Timestamp(EVAL_END))
    pilot = set(pd.read_csv(BASE / "age2030_pilot" / "aligned_panel_40.csv", dtype={"admin_dong_code": str}).admin_dong_code)
    for horizon, method, lag in specs:
        if method == "trailing_7day_mean":
            pred = sum(y.groupby(level=0).shift(lag + j) for j in range(7)) / 7
            pred = pred.loc[test]
        else:
            pred = y.groupby(level=0).shift(lag).loc[test]
        actual = y.loc[test]
        valid = actual.notna() & pred.notna()
        part = pd.DataFrame({"date": actual.index.get_level_values("date")[valid].strftime("%Y-%m-%d"),
                             "admin_dong_code": all_dongs[test][valid],
                             "actual": actual[valid].to_numpy(), "predicted": pred[valid].to_numpy()})
        part["horizon_days"] = horizon
        part["method"] = method
        part["lag_days"] = lag
        part["pilot_40"] = part.admin_dong_code.isin(pilot)
        outputs.append(part)
    predictions = pd.concat(outputs, ignore_index=True)
    required = pd.Series({h: sum(s[0] == h for s in specs) for h in {s[0] for s in specs}})
    present = predictions.groupby(["horizon_days", "admin_dong_code", "date"]).method.transform("nunique")
    predictions = predictions.loc[present.eq(predictions.horizon_days.map(required))].copy()
    predictions.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    result = []
    for scope, p in (("all_dongs", predictions), ("pilot_40", predictions.loc[predictions.pilot_40])):
        for (horizon, method), g in p.groupby(["horizon_days", "method"], sort=True):
            err = (g.actual - g.predicted).abs()
            result.append({"scope": scope, "horizon_days": int(horizon), "method": method,
                           "n": len(g), "dongs": int(g.admin_dong_code.nunique()),
                           "mae": round(float(err.mean()), 2),
                           "wape_pct": round(float(100 * err.sum() / g.actual.sum()), 2)})
    metrics = pd.DataFrame(result)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    coverage = {"months": len(files), "dates": int(panel.date.nunique()),
                "dong_codes": int(panel.admin_dong_code.nunique()), "daily_rows": len(panel),
                "max_monthly_reaggregation_difference": max_monthly_difference,
                "missing_after_calendar_reindex": int(y.isna().sum()),
                "test": f"{EVAL_START} to {EVAL_END}",
                "baseline_note": "91-day origin approximates 3 calendar months; not an exact month offset"}
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return coverage


if __name__ == "__main__":
    months = [f"{year}{month:02d}" for year in range(2023, 2026) for month in range(1, 13)
              if f"{year}{month:02d}" <= "202508"]
    audit = []
    for month in months:
        info = aggregate_month(month)
        audit.append(info)
        print(f"{month}: {info['days']} dates, {info['rows']} rows", flush=True)
    (OUT / "aggregation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(evaluate(), indent=2), flush=True)
