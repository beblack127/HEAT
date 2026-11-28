"""Journal-style figure for the subway EDA, using the Seoul open-data CSVs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subway_eda import load


ROOT = Path(__file__).resolve().parent


def main():
    font_name = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })

    d = load()
    d = d[d["호선명"] == "2호선"].copy()
    daily = d.groupby(["date", "역명"], as_index=False)["하차총승객수"].sum()
    daily["year"] = daily.date.dt.year
    daily["month"] = daily.date.dt.month

    annual = daily.groupby(["역명", "year"]).agg(
        days=("date", "nunique"), mean_exits=("하차총승객수", "mean")
    ).reset_index()
    means = annual.pivot(index="역명", columns="year", values="mean_exits")
    days = annual.pivot(index="역명", columns="year", values="days")
    eligible = (days[2024] >= 350) & (days[2025] >= 350) & (means[2024] >= 3000)
    means = means.loc[eligible].copy()
    means["yoy_pct"] = (means[2025] / means[2024] - 1) * 100
    means["rank"] = means["yoy_pct"].rank(ascending=False, method="min")
    n = len(means)

    monthly = daily[daily["역명"].isin(means.index)].groupby(
        ["역명", "year", "month"], as_index=False
    )["하차총승객수"].mean()
    monthly_wide = monthly.pivot(index=["역명", "month"], columns="year", values="하차총승객수")
    monthly_wide["yoy_pct"] = (monthly_wide[2025] / monthly_wide[2024] - 1) * 100
    month_summary = monthly_wide.groupby("month")["yoy_pct"].agg(
        mean="mean"
    )
    seongsu = monthly_wide.xs("성수", level="역명")["yoy_pct"]
    hongdae = monthly_wide.xs("홍대입구", level="역명")["yoy_pct"]

    RED, BLUE = "#C43C39", "#2F6DAA"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": .37})
    fig.patch.set_facecolor("white")

    # (a) Cross-sectional growth against the initial station size.
    other = means.drop(index=["성수", "홍대입구"])
    ax1.scatter(other[2024] / 1000, other["yoy_pct"], s=13,
                facecolor="white", edgecolor="0.40", linewidth=.6, zorder=3)
    s = means.loc["성수"]
    ax1.scatter([s[2024] / 1000], [s["yoy_pct"]], s=34,
                facecolor="black", edgecolor="black", zorder=4, label="Seongsu")
    h = means.loc["홍대입구"]
    ax1.scatter([h[2024] / 1000], [h["yoy_pct"]], s=38, marker="s",
                facecolor=RED, edgecolor=RED, zorder=4, label="Hongdae (Hongik Univ.)")
    mean_size = means[2024].mean() / 1000
    mean_growth = means["yoy_pct"].mean()
    ax1.scatter([mean_size], [mean_growth], s=52, marker="D",
                facecolor=BLUE, edgecolor=BLUE, zorder=5, label="Line 2 station mean")
    ax1.annotate(f"Seongsu ({int(s['rank'])}/{n})", (s[2024] / 1000, s["yoy_pct"]),
                 xytext=(5, -4), textcoords="offset points", fontsize=6.5, ha="left", va="top")
    ax1.axhline(0, color="0.45", lw=.6, ls="--", zorder=1)
    ax1.set_xlabel("Mean daily alightings in 2024 (thousands)")
    ax1.set_ylabel("Annual change in 2025 (%)")
    ax1.set_xlim(0, 85)
    ax1.set_ylim(-11, 19)
    ax1.set_xticks([0, 20, 40, 60, 80])
    ax1.set_yticks([-10, 0, 10])
    ax1.set_title("(a) Station size and annual change", loc="left", fontsize=9, pad=7)

    # (b) Monthly year-on-year trajectories for the two stations and the
    # unweighted mean across all 46 eligible Line 2 stations.
    x = month_summary.index.to_numpy(dtype=int)
    ax2.plot(x, month_summary["mean"].to_numpy(), color=BLUE, lw=1.25,
             marker="D", markersize=2.5, label="Line 2 station mean")
    ax2.plot(hongdae.index.to_numpy(dtype=int), hongdae.to_numpy(), color=RED, lw=1.25,
             marker="s", markersize=2.5, label="Hongdae (Hongik Univ.)")
    ax2.plot(seongsu.index.to_numpy(dtype=int), seongsu.to_numpy(), color="black", lw=1.4,
             marker="o", markersize=2.8, label="Seongsu")
    ax2.axhline(0, color="0.45", lw=.6, ls=":", zorder=1)
    ax2.set_xlabel("Month in 2025")
    ax2.set_ylabel("Change from same month in 2024 (%)")
    ax2.set_xlim(.8, 12.2)
    ax2.set_xticks([1, 3, 5, 7, 9, 11])
    ax2.set_title("(b) Monthly year-on-year change", loc="left", fontsize=9, pad=7)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.8)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.grid(axis="y", color="0.92", lw=.45)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.89)
    out_png = ROOT / "subway_eda_english.png"
    out_svg = ROOT / "subway_eda_english.svg"
    fig.savefig(out_png, dpi=400, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    plt.close(fig)
    print(f"stations={n}, seongsu_rank={int(s['rank'])}, seongsu_yoy={s['yoy_pct']:.1f}%")
    print(f"hongdae_yoy={h['yoy_pct']:.1f}%, station_mean_growth={mean_growth:.1f}%")
    print(f"saved: {out_png}")
    print(f"saved: {out_svg}")


if __name__ == "__main__":
    main()
