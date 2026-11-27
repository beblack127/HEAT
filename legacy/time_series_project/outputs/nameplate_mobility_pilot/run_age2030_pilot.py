"""40-dong two-step forecast pilot for 20s+30s destination mobility 4+5+7."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import search_mobility_pilot as old

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_pilot"
OUT.mkdir(parents=True, exist_ok=True)
AGE = ROOT / "outputs" / "nameplate_mobility_pilot" / "age2030_mobility"
SUBWAY = ROOT / "work" / "Time-series-Project" / "data" / "context_monthly" / "subway_station_monthly_2023_2025.csv"
STATION_BRIDGE = ROOT / "work" / "HEAT" / "data" / "processed" / "nametags" / "05_행정동_역_연결표.csv"


def load_age() -> pd.DataFrame:
    files = sorted(AGE.glob("age2030_457_20????.csv"))
    if len(files) != 32:
        raise ValueError(f"Expected 32 age-aggregate files through 2025-08; found {len(files)}")
    age = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str}) for p in files], ignore_index=True)
    if age.duplicated(["admin_dong_code", "month"]).any():
        raise ValueError("Duplicate age target")
    return age


def baselines(age: pd.DataFrame, pilot_codes: set[str]) -> pd.DataFrame:
    df = age.sort_values(["admin_dong_code", "month"]).copy()
    df["lag2"] = df.groupby("admin_dong_code").mobility_457_age20_39_daily.shift(2)
    df["lag12"] = df.groupby("admin_dong_code").mobility_457_age20_39_daily.shift(12)
    test = df.loc[df.month.ge("2025-01") & df.month.le("2025-08") & df.admin_dong_code.isin(pilot_codes)].copy()
    if len(test) != 320 or test[["lag2", "lag12"]].isna().any().any():
        raise ValueError("Incomplete common baseline evaluation panel")
    parts = []
    for name, col in (("lag2", "lag2"), ("lag12", "lag12")):
        part = test[["admin_dong_code", "month", "mobility_457_age20_39_daily", col]].copy()
        part = part.rename(columns={"mobility_457_age20_39_daily": "actual_daily", col: "predicted_daily"})
        part["model"] = name
        part["abs_error"] = (part.actual_daily - part.predicted_daily).abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def citywide_baselines(age: pd.DataFrame) -> pd.DataFrame:
    """Simple benchmarks for all dongs with an observed 12-month history."""
    df = age.sort_values(["admin_dong_code", "month"]).copy()
    df["lag2"] = df.groupby("admin_dong_code").mobility_457_age20_39_daily.shift(2)
    df["lag12"] = df.groupby("admin_dong_code").mobility_457_age20_39_daily.shift(12)
    test = df.loc[df.month.ge("2025-01") & df.month.le("2025-08")].dropna(subset=["lag2", "lag12"])
    parts = []
    for name, col in (("lag2", "lag2"), ("lag12", "lag12")):
        part = test[["admin_dong_code", "month", "mobility_457_age20_39_daily", col]].copy()
        part = part.rename(columns={"mobility_457_age20_39_daily": "actual_daily", col: "predicted_daily"})
        part["model"] = name
        part["abs_error"] = (part.actual_daily - part.predicted_daily).abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def nearest_station_feature(codes: set[str]) -> pd.DataFrame:
    """Nearby station exits: fixed closest-station mapping, two-month lag."""
    bridge = pd.read_csv(STATION_BRIDGE, encoding="utf-8-sig", dtype={"dong_code8": str})
    station = pd.read_csv(SUBWAY, encoding="utf-8-sig")
    station = station.loc[station.month.le("2025-08")]
    station = station.groupby(["station_name", "month"], as_index=False).agg(exits=("exits", "sum"), days=("days", "max"))
    complete_names = station.groupby("station_name").month.nunique()
    complete_names = set(complete_names.index[complete_names.eq(32)])
    bridge = bridge.loc[bridge.dong_code8.isin(codes) & bridge.station.isin(complete_names)]
    bridge = bridge.sort_values(["dong_code8", "dist_to_dong_m", "station"])
    bridge = bridge.drop_duplicates("dong_code8")[["dong_code8", "station", "dist_to_dong_m"]]
    if len(bridge) != len(codes):
        raise ValueError("No stable nearby station for some pilot dongs")
    station["exits_daily"] = station.exits / station.days
    frame = bridge.merge(station, left_on="station", right_on="station_name", how="left", validate="many_to_many")
    if frame.duplicated(["dong_code8", "month"]).any():
        raise ValueError("Duplicate nearby station month")
    frame = frame.rename(columns={"dong_code8": "admin_dong_code"}).sort_values(["admin_dong_code", "month"])
    frame["subway_lag2"] = frame.groupby("admin_dong_code").exits_daily.shift(2)
    return frame[["admin_dong_code", "month", "station", "dist_to_dong_m", "subway_lag2"]]


def two_stage_search(dong: pd.DataFrame) -> tuple[list[dict], dict]:
    """Fit y~current search; forecast unavailable search by last known search."""
    train = dong.loc[dong.month.le(old.TRAIN_END)].copy()
    mean, std = float(train.search_current_x.mean()), float(train.search_current_x.std(ddof=0))
    std = std if std > 1e-8 else 1.0
    scaled = dong.copy()
    scaled["search_current_scaled"] = (scaled.search_current_x - mean) / std
    train = scaled.loc[scaled.month.le(old.TRAIN_END)]
    model = SARIMAX(train.log_y.to_numpy(), exog=train[["search_current_scaled"]].to_numpy(),
                    order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), trend="n",
                    enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=200)
        optimizer = "lbfgs"
        if not result.mle_retvals.get("converged", False):
            result = model.fit(method="powell", disp=False, maxiter=300)
            optimizer = "powell"
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("two_stage_search fit nonconverged")
    rows = []
    state_last = old.TRAIN_END
    for _, row in scaled.loc[scaled.month.ge(old.TEST_START) & scaled.month.le(old.TEST_END)].iterrows():
        future = scaled.loc[scaled.month.gt(state_last) & scaled.month.le(row.month)]
        if len(future) != 2:
            raise ValueError("Expected two-step window")
        last_x = float(scaled.loc[scaled.month.eq(state_last), "search_current_scaled"].iloc[0])
        pred_search = np.array([[last_x], [last_x]])
        predicted = max(0.0, float(np.expm1(min(float(result.forecast(steps=2, exog=pred_search)[-1]), 20))))
        rows.append({"admin_dong_code": row.admin_dong_code, "month": row.month,
                     "model": "ARIMAX_search_forecast", "actual_daily": float(row.mobility_shop_tour_other_daily),
                     "predicted_daily": predicted, "abs_error": abs(float(row.mobility_shop_tour_other_daily) - predicted)})
        available = future.iloc[0]
        result = result.append([available.log_y], exog=np.array([[available.search_current_scaled]]), refit=False)
        state_last = available.month
    return rows, {"optimizer": optimizer, "warnings": len(caught)}


def main() -> None:
    age = load_age()
    original, coverage = old.load_panel("share")
    pilot_codes = set(original.admin_dong_code)
    baseline = baselines(age, pilot_codes)
    city_baseline = citywide_baselines(age)
    city_baseline.to_csv(OUT / "citywide_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    age_pilot = age.loc[age.admin_dong_code.isin(pilot_codes), ["admin_dong_code", "month", "mobility_457_age20_39_daily", "mobility_457_allages_daily", "age20_39_share"]]
    panel = original.merge(age_pilot, on=["admin_dong_code", "month"], how="left", validate="one_to_one")
    panel = panel.merge(nearest_station_feature(pilot_codes), on=["admin_dong_code", "month"], how="left", validate="one_to_one")
    if panel.mobility_457_age20_39_daily.isna().any():
        raise ValueError("Age target missing for pilot")
    if not np.allclose(panel.mobility_shop_tour_other_daily, panel.mobility_457_allages_daily, rtol=0, atol=1e-5):
        raise ValueError("Old all-age target does not match raw reaggregation")
    panel["mobility_shop_tour_other_daily"] = panel.mobility_457_age20_39_daily
    panel["log_y"] = np.log1p(panel.mobility_457_age20_39_daily)
    panel["search_current_x"] = np.log1p(1_000_000 * panel.search_share)
    panel["subway_x"] = np.log1p(panel.subway_lag2)
    if panel.subway_x.isna().any():
        raise ValueError("Missing nearby station feature")
    panel = panel.sort_values(["admin_dong_code", "month"])
    panel.to_csv(OUT / "aligned_panel_40.csv", index=False, encoding="utf-8-sig")
    selected = {"ARIMA": old.SPECS["ARIMA"], "SARIMA": old.SPECS["SARIMA"],
                "ARIMAX_search": old.SPECS["ARIMAX_search"], "SARIMAX_search": old.SPECS["SARIMAX_search"],
                "ARIMAX_living": old.SPECS["ARIMAX_living"], "ARIMAX_both": old.SPECS["ARIMAX_both"],
                "ARIMAX_subway": ("subway_x",), "SARIMAX_subway": ("subway_x",)}
    all_pred = [baseline]
    fit_audit = []
    for i, code in enumerate(sorted(pilot_codes), 1):
        dong = panel.loc[panel.admin_dong_code.eq(code)].copy()
        rows = []
        for name, features in selected.items():
            p, audit = old.fit_predict_one(dong, name, features)
            rows.extend(p)
            fit_audit.append({"admin_dong_code": code, "model": name, **audit})
        p, audit = two_stage_search(dong)
        rows.extend(p)
        fit_audit.append({"admin_dong_code": code, "model": "ARIMAX_search_forecast", **audit})
        all_pred.append(pd.DataFrame(rows))
        if i % 5 == 0 or i == len(pilot_codes):
            print(f"modeled {i}/{len(pilot_codes)} dongs", flush=True)
    pred = pd.concat(all_pred, ignore_index=True)
    if pred.groupby(["admin_dong_code", "month"]).model.nunique().ne(11).any():
        raise ValueError("Models evaluated on unequal rows")
    pred.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit_audit).to_csv(OUT / "fit_audit.csv", index=False, encoding="utf-8-sig")
    summary = {"target": "20s+30s destination mobility, purposes 4+5+7, daily average",
               "period": "2023-01 to 2025-08", "train": "2023-03 to 2024-11", "test": "2025-01 to 2025-08",
               "evaluation": "40 dongs x 8 months, 2-step forecast from m-2",
               "two_stage": "ARIMAX fit with contemporaneous search; search for m-1 and m predicted at m-2 by last observed value",
               "subway": "closest HEAT-mapped station with complete 2023-01 to 2025-08 history; all line exits summed by name; two-month lag; spatial proxy only",
               "coverage": coverage,
               "target_mean_age_share_40": float(panel.age20_39_share.mean()),
               "citywide_baselines": {name: old.score(g) for name, g in city_baseline.groupby("model")},
               "models": {name: old.score(g) for name, g in pred.groupby("model")},
               "caveat": "Exploratory pilot; fixed orders; source anomaly 2024-08 to 2024-10 in training; same test used in prior work"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["models"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
