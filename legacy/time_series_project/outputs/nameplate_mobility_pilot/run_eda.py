"""EDA for the 40-dong mobility pilot and the supplied nameplate ZIP.

python run_eda.py --repo PATH_TO_REPO --nameplate PATH_TO_ZIP --out OUTPUT_DIR
Requires pandas, numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def read_files(folder: Path, stem: str) -> pd.DataFrame:
    files = sorted(folder.glob(f"{stem}_20????.csv"))
    if len(files) != 36:
        raise ValueError(f"Expected 36 {stem} files, got {len(files)}")
    return pd.concat((pd.read_csv(p, encoding="utf-8-sig", dtype={"admin_dong_code": str})
                      for p in files), ignore_index=True)


def read_zip_table(archive: zipfile.ZipFile, prefix: str) -> pd.DataFrame:
    names = [n for n in archive.namelist() if n.split("/")[-1].startswith(prefix)
             and n.lower().endswith(".csv") and not n.startswith("__MACOSX/") and "/._" not in n]
    if len(names) != 1:
        raise ValueError(f"Expected one {prefix} table")
    return pd.read_csv(io.BytesIO(archive.read(names[0])), encoding="utf-8-sig", dtype={"dong_code8": str})


def numeric_summary(series: pd.Series) -> dict:
    q = series.quantile([0, .25, .5, .75, .9, .95, 1]).to_dict()
    return {"min": round(float(q[0]), 2), "q25": round(float(q[.25]), 2),
            "median": round(float(q[.5]), 2), "q75": round(float(q[.75]), 2),
            "q90": round(float(q[.9]), 2), "q95": round(float(q[.95]), 2),
            "max": round(float(q[1]), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--nameplate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folder = args.repo / "data" / "context_monthly"
    mob, living = read_files(folder, "mobility"), read_files(folder, "living")
    for label, df in (("mobility", mob), ("living", living)):
        if df.duplicated(["admin_dong_code", "month"]).any():
            raise ValueError(f"Duplicate {label} dong-month")
    with zipfile.ZipFile(args.nameplate) as z:
        names = read_zip_table(z, "01_")
        links = read_zip_table(z, "04_")
    names = names[["dong_code8", "gu", "dong_name", "n_동네", "n_역", "n_권역"]]
    df = mob.merge(living[["admin_dong_code", "month", "living_day_mean", "living_weekend_day_mean"]],
                   on=["admin_dong_code", "month"], how="left", validate="one_to_one")
    df = df.merge(names, left_on="admin_dong_code", right_on="dong_code8", how="left", validate="many_to_one")
    df = df.sort_values(["admin_dong_code", "month"]).reset_index(drop=True)
    df["year"] = df["month"].str[:4].astype(int)
    df["month_num"] = df["month"].str[5:7].astype(int)
    df["target_daily"] = df["mobility_shop_tour_other_daily"]
    df["shop_tour_daily"] = df["mobility_shop_tour_daily"]
    df["other_daily"] = df["target_daily"] - df["shop_tour_daily"]
    df["target_share_all"] = df["target_daily"] / df["mobility_all_arrivals_daily"]
    df["shop_tour_share_target"] = df["shop_tour_daily"] / df["target_daily"]
    df["target_yoy"] = df.groupby("admin_dong_code")["target_daily"].pct_change(12, fill_method=None)
    df["target_mom"] = df.groupby("admin_dong_code")["target_daily"].pct_change(fill_method=None)
    df["log_target_change"] = df.groupby("admin_dong_code")["target_daily"].transform(lambda s: np.log(s).diff())
    df["log_living_change"] = df.groupby("admin_dong_code")["living_day_mean"].transform(lambda s: np.log(s).diff())

    expected = pd.MultiIndex.from_product([sorted(df.admin_dong_code.unique()),
                                           pd.period_range("2023-01", "2025-12", freq="M").astype(str)],
                                          names=["admin_dong_code", "month"])
    missing_pairs = expected.difference(pd.MultiIndex.from_frame(df[["admin_dong_code", "month"]]))
    monthly = df.groupby("month", as_index=False).agg(total_target_daily=("target_daily", "sum"),
        total_shop_tour_daily=("shop_tour_daily", "sum"), total_all_daily=("mobility_all_arrivals_daily", "sum"),
        median_dong_target=("target_daily", "median"), zero_shop_tour=("shop_tour_daily", lambda s: int((s == 0).sum())))
    monthly["year"] = monthly.month.str[:4].astype(int)
    monthly["month_num"] = monthly.month.str[5:7].astype(int)
    monthly["total_yoy"] = monthly.total_target_daily.pct_change(12, fill_method=None)
    monthly.to_csv(args.out / "eda_monthly.csv", index=False, encoding="utf-8-sig")
    dong = df.groupby(["admin_dong_code", "gu", "dong_name"], as_index=False).agg(
        mean_target_daily=("target_daily", "mean"), median_target_daily=("target_daily", "median"),
        mean_all_daily=("mobility_all_arrivals_daily", "mean"),
        shop_tour_nonzero_months=("shop_tour_daily", lambda s: int((s > 0).sum())),
        mean_shop_tour_share=("shop_tour_share_target", "mean"),
        sd_target_daily=("target_daily", "std"))
    dong["cv_target"] = dong.sd_target_daily / dong.mean_target_daily
    dong.to_csv(args.out / "eda_dong_summary.csv", index=False, encoding="utf-8-sig")

    yearly = df.groupby("year").agg(mean_total_target=("target_daily", "sum"),
        total_shop_tour=("shop_tour_daily", "sum"), total_all=("mobility_all_arrivals_daily", "sum"))
    yearly["mean_total_target"] /= 12
    yoy = df[df.year >= 2024].copy()
    corr_level = df[["target_daily", "living_day_mean"]].corr().iloc[0, 1]
    corr_log_change = df[["log_target_change", "log_living_change"]].dropna().corr().iloc[0, 1]
    autocorr = []
    for code, group in df.groupby("admin_dong_code"):
        values = group.target_daily.reset_index(drop=True)
        autocorr.append({"admin_dong_code": code, "lag1": values.autocorr(1), "lag12": values.autocorr(12)})
    ac = pd.DataFrame(autocorr)
    ac.to_csv(args.out / "eda_autocorrelation.csv", index=False, encoding="utf-8-sig")
    anomalous_months = {"2024-08", "2024-09", "2024-10"}
    lag12_without_anomaly = []
    for _, group in df.groupby("admin_dong_code"):
        group = group.reset_index(drop=True)
        pairs = [(group.iloc[i-12].target_daily, group.iloc[i].target_daily)
                 for i in range(12, len(group))
                 if group.iloc[i-12]["month"] not in anomalous_months
                 and group.iloc[i]["month"] not in anomalous_months]
        lag12_without_anomaly.append(float(np.corrcoef(np.array(pairs).T)[0, 1]))
    non_anomaly = df[~df.month_num.isin([8, 9, 10])].groupby("year").target_daily.sum()
    aug_oct = df[df.month_num.isin([8, 9, 10])].groupby("year").target_daily.sum()
    living_monthly = df.groupby("month")["living_day_mean"].sum()

    zero = df.shop_tour_daily.eq(0)
    top = dong.nlargest(5, "mean_target_daily")
    result = {
        "data": {"rows": len(df), "dongs": df.admin_dong_code.nunique(), "months": df.month.nunique(),
            "missing_dong_month_pairs": len(missing_pairs), "duplicate_pairs": int(df.duplicated(["admin_dong_code", "month"]).sum()),
            "missing_living": int(df.living_day_mean.isna().sum()), "unmatched_nameplate": int(df.dong_name.isna().sum()),
            "nameplate_dongs": len(names), "sample_dongs_with_region_slot": int((dong.admin_dong_code.map(
                df.groupby("admin_dong_code")["n_권역"].first()) > 0).sum()),
            "days_values": sorted(df.days.unique().tolist()),
            "invalid_target_rows": int(((df.target_daily < 0) | (df.target_daily > df.mobility_all_arrivals_daily)).sum())},
        "target": {"daily_distribution": numeric_summary(df.target_daily),
            "weighted_share_of_all": round(float(df.target_daily.sum() / df.mobility_all_arrivals_daily.sum()), 4),
            "shop_tour_share_of_target": round(float(df.shop_tour_daily.sum() / df.target_daily.sum()), 4),
            "shop_tour_zero_rows": int(zero.sum()), "shop_tour_zero_pct": round(float(zero.mean()), 4),
            "dongs_shop_tour_all_zero": int((dong.shop_tour_nonzero_months == 0).sum()),
            "other_negative_rows": int((df.other_daily < -1e-8).sum())},
        "yearly_mean_total_daily": {str(y): round(float(v), 2) for y, v in yearly.mean_total_target.items()},
        "yearly_change_pct": {"2024_vs_2023": round(float(yearly.loc[2024, "mean_total_target"] /
            yearly.loc[2023, "mean_total_target"] - 1), 4), "2025_vs_2024": round(float(yearly.loc[2025, "mean_total_target"] /
            yearly.loc[2024, "mean_total_target"] - 1), 4)},
        "monthly_yoy": {"median_2025": round(float(yoy[yoy.year == 2025].target_yoy.median()), 4),
            "q10_2025": round(float(yoy[yoy.year == 2025].target_yoy.quantile(.1)), 4),
            "q90_2025": round(float(yoy[yoy.year == 2025].target_yoy.quantile(.9)), 4),
            "rows_abs_gt_50pct_2025": int((yoy[yoy.year == 2025].target_yoy.abs() > .5).sum())},
        "monthly_extremes": {"highest_total": monthly.nlargest(3, "total_target_daily")[["month", "total_target_daily"]].to_dict("records"),
            "lowest_total": monthly.nsmallest(3, "total_target_daily")[["month", "total_target_daily"]].to_dict("records"),
            "highest_yoy": monthly.nlargest(3, "total_yoy")[["month", "total_yoy"]].to_dict("records"),
            "lowest_yoy": monthly.nsmallest(3, "total_yoy")[["month", "total_yoy"]].to_dict("records")},
        "top_dongs": top[["gu", "dong_name", "mean_target_daily"]].to_dict("records"),
        "top5_share_target": round(float(top.mean_target_daily.sum() / dong.mean_target_daily.sum()), 4),
        "autocorrelation_median": {"lag1": round(float(ac.lag1.median()), 4), "lag12": round(float(ac.lag12.median()), 4)},
        "anomaly_2024_aug_oct": {
            "august_dongs_yoy_gt_50pct": int((df[df.month == "2024-08"].target_yoy > .5).sum()),
            "aug_oct_total_vs_2023": round(float(aug_oct.loc[2024] / aug_oct.loc[2023] - 1), 4),
            "aug_oct_2025_vs_2023": round(float(aug_oct.loc[2025] / aug_oct.loc[2023] - 1), 4),
            "non_aug_oct_2024_vs_2023": round(float(non_anomaly.loc[2024] / non_anomaly.loc[2023] - 1), 4),
            "non_aug_oct_2025_vs_2024": round(float(non_anomaly.loc[2025] / non_anomaly.loc[2024] - 1), 4),
            "living_2024_yoy": {f"{m:02d}": round(float(living_monthly.loc[f"2024-{m:02d}"] /
                living_monthly.loc[f"2023-{m:02d}"] - 1), 4) for m in (8, 9, 10)},
            "lag12_autocorr_median_excluding_affected_pairs": round(float(np.nanmedian(lag12_without_anomaly)), 4)},
        "living_correlation": {"pooled_levels": round(float(corr_level), 4),
            "pooled_within_dong_log_changes": round(float(corr_log_change), 4)},
        "nameplate": {"usable_links": int((links.usable == 1).sum()),
            "usable_name_dongs": int(links[links.usable == 1].dong_code8.nunique())}}
    (args.out / "eda_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    font = "Malgun Gothic" if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    months = np.arange(len(monthly))
    axes[0, 0].plot(months, monthly.total_target_daily / 1000, color="#1f6d9c", lw=2.3)
    axes[0, 0].set(title="40개 동의 일평균 도착 이동량 합계", ylabel="천 건/일")
    axes[0, 0].set_xticks([0, 6, 12, 18, 24, 30, 35], monthly.month.iloc[[0, 6, 12, 18, 24, 30, 35]], rotation=30)
    axes[0, 1].boxplot([df[df.year == y].target_daily / 1000 for y in (2023, 2024, 2025)],
                       tick_labels=["2023", "2024", "2025"], showfliers=False)
    axes[0, 1].set(title="동별 월 유입량 분포", ylabel="천 건/일")
    axes[1, 0].plot(months, monthly.zero_shop_tour, color="#b35c44", lw=2.3)
    axes[1, 0].set(title="쇼핑·관광(4+5)이 0인 동 수", ylabel="40개 동 중", ylim=(0, 40))
    axes[1, 0].set_xticks([0, 6, 12, 18, 24, 30, 35], monthly.month.iloc[[0, 6, 12, 18, 24, 30, 35]], rotation=30)
    axes[1, 1].hist(ac.lag1.dropna(), bins=12, alpha=.7, color="#2c8c72", label="1개월")
    axes[1, 1].hist(ac.lag12.dropna(), bins=12, alpha=.55, color="#d4943c", label="12개월")
    axes[1, 1].set(title="행정동별 자기상관 분포", xlabel="상관계수", ylabel="동 수")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(color="#dbe3eb", alpha=.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8fafc")
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("생활이동 파일럿 EDA · 2023–2025 · 행정동 40개", fontsize=21, weight="bold")
    fig.text(.5, .015, "목표값: 목적 4+5+7 일평균 도착 이동량 · 통신 기반 추정치 · 표본은 서울 전체를 대표하지 않음",
             ha="center", fontsize=10, color="#536579")
    fig.tight_layout(rect=[.02, .04, .98, .93])
    fig.savefig(args.out / "eda_overview.png", facecolor=fig.get_facecolor())

    fig2, ax2 = plt.subplots(figsize=(13, 5.5), dpi=150)
    fig2.patch.set_facecolor("#f8fafc")
    positions = np.arange(1, 13)
    def yoy_values(series):
        return np.array([(series.loc[f"2024-{m:02d}"] / series.loc[f"2023-{m:02d}"] - 1) * 100
                         for m in positions])
    ax2.plot(positions, yoy_values(monthly.set_index("month").total_target_daily),
             marker="o", lw=2.7, color="#176b9a", label="목적 4+5+7 이동량")
    ax2.plot(positions, yoy_values(monthly.set_index("month").total_all_daily),
             marker="s", lw=2.1, color="#a8663a", label="전체 목적 이동량")
    ax2.plot(positions, yoy_values(living_monthly), marker="^", lw=2.1,
             color="#35885d", label="주간 생활인구")
    ax2.axhline(0, color="#8b99a9", lw=1)
    ax2.axvspan(7.5, 10.5, color="#f7d0c1", alpha=.5)
    ax2.set(xlim=(1, 12), xticks=positions, xlabel="2024년 월", ylabel="2023년 같은 달 대비 증감률 (%)",
            title="2024년 8–10월 생활이동 급증: 생활인구는 같은 폭으로 증가하지 않음")
    ax2.grid(color="#dbe3eb"); ax2.set_axisbelow(True); ax2.set_facecolor("#f8fafc")
    ax2.legend(frameon=False, ncol=3, loc="upper left")
    for s in ax2.spines.values(): s.set_visible(False)
    fig2.text(.5, .02, "40개 동 합계 기준 · 급증의 원인은 현재 미확인 · 원본 파일과 재배포 이력 대조 필요",
              ha="center", fontsize=10, color="#6e4750")
    fig2.tight_layout(rect=[.02, .08, .98, .98])
    fig2.savefig(args.out / "eda_2024_anomaly.png", facecolor=fig2.get_facecolor())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
