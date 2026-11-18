"""11쪽 'Why it worked' 왼쪽 그림: 네 요소(보정·계절성·최근 수준·평균)를 작은 그림 4개로.
출력: reports/gt_midterm/slides/fig/S13_why_{1..4}.png, S13_why_grid.png
"""
import glob, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import config as C

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = C.REPORT / "slides" / "fig"; OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(pathlib.Path.home() / ".fonts/KaiseiTokumin-*.ttf")): fm.fontManager.addfont(f)
BG, NAVY, RED, INK, GREY, LIGHT = "#F7F7F7", "#004490", "#AE3A1E", "#222222", "#8A8F98", "#C9CED6"
GREEN = "#1A9E6E"
plt.rcParams.update({"font.family": "Kaisei Tokumin", "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 18, "xtick.labelsize": 17, "ytick.labelsize": 17,
                     "legend.fontsize": 16, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "savefig.dpi": 200})

T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
W = pd.read_parquet(ROOT / "results/gt/wape_by_origin.parquet")
W["o"] = pd.to_datetime(W.origin).dt.strftime("%y.%m")
wp = W.pivot(index="key", columns="o", values="wape")
ORI = list(wp.columns)
fac = pd.read_csv(ROOT / "results/gt/clean_factor.csv", parse_dates=["week_end"]).set_index("week_end").factor
units = set(pd.read_parquet(ROOT / "results/gt/pred_comb_2025-04-27.parquet").unit)


def p1(ax):
    """Cleaning: 작년 같은 주 예측 오차, 원자료 vs 보정 (4시점 평균, 2025.07 시점)"""
    raw, cl = wp.loc["snaive|none"], wp.loc["snaive_c|none"]
    x = np.arange(2); w = .36
    a = ax.bar(x - w / 2, [raw.mean(), raw["25.07"]], w, color=LIGHT, label="Raw")
    b = ax.bar(x + w / 2, [cl.mean(), cl["25.07"]], w, color=NAVY, label="Cleaned")
    for r in list(a) + list(b): ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.1f}%", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, ["4-date avg", "Jul 2025 date"]); ax.set_ylim(0, 60); ax.set_ylabel("WAPE (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Cleaning", loc="left", color=NAVY, fontweight="bold")
    ax.text(0.99, 0.97, "forecast = same week last year", transform=ax.transAxes, ha="right", va="top", fontsize=15, color=GREY)


def p2(ax):
    """Season: 계절 없음(지난주 값) vs 계절 있음(작년 같은 주), 이상 구간이 안 걸린 2시점 평균"""
    cl = ["25.01", "25.04"]
    v = [wp.loc["naive|none", cl].mean(), wp.loc["snaive_c|none", cl].mean()]
    x = np.arange(2)
    bars = ax.bar(x, v, .55, color=[LIGHT, NAVY])
    for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.1f}%", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, ["Last week\n(no season)", "Same week last year\n(season)"]); ax.set_ylim(0, 30); ax.set_ylabel("WAPE (%)")
    ax.set_title("Seasonality", loc="left", color=NAVY, fontweight="bold")
    ax.text(0.99, 0.97, "2 clean dates (25.01, 25.04)", transform=ax.transAxes, ha="right", va="top", fontsize=15, color=GREY)


def p3(ax):
    """Recent level: 동별 수준 변화 (2025 상반기 vs 2024 상반기)"""
    d = T[T.unit.isin(units)]
    a = d[(d.week_end >= "2024-01-01") & (d.week_end < "2024-07-01")].groupby("unit").y.mean()
    b = d[(d.week_end >= "2025-01-01") & (d.week_end < "2025-07-01")].groupby("unit").y.mean()
    r = ((b / a - 1) * 100).clip(-40, 60)
    bins = np.arange(-40, 62.5, 5)
    n, e, patches = ax.hist(r, bins=bins, color=NAVY, edgecolor=BG)
    for p, lo in zip(patches, e[:-1]):
        if -10 <= lo < 10: p.set_facecolor(LIGHT)
    share = (((b / a - 1) * 100).abs() > 10).mean() * 100
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Change vs last year (%), Jan–Jun"); ax.set_ylabel("Dongs")
    ax.set_title("Recent level", loc="left", color=NAVY, fontweight="bold")
    ax.text(0.99, 0.97, f"{share:.0f}% of dongs\nmoved more than ±10%", transform=ax.transAxes, ha="right", va="top", fontsize=17, color=NAVY, fontweight="bold")


def p4(ax):
    """Averaging: 시점별 오차, 서로 다른 시점에 틀림"""
    x = np.arange(len(ORI))
    ser = [("Last year (cleaned)", "snaive_c|none", GREY, "o", 2.2), ("ARIMA + Fourier", "arima_fourier|none", GREEN, "s", 2.2), ("Average (ours)", "comb_fourier|none", RED, "D", 3.4)]
    for lab, k, c, m, lw in ser:
        v = wp.loc[k, ORI].values
        ax.plot(x, v, color=c, marker=m, ms=9, lw=lw, label=f"{lab}  {v.mean():.1f}")
    ax.set_xticks(x, ORI); ax.set_ylim(4, 27); ax.set_ylabel("WAPE (%)"); ax.set_xlabel("Forecast date")
    ax.legend(frameon=False, loc="upper left", title="mean WAPE (%)", title_fontsize=15, alignment="left")
    ax.set_title("Averaging", loc="left", color=NAVY, fontweight="bold")


panels = [p1, p2, p3, p4]
for i, p in enumerate(panels, 1):
    f, ax = plt.subplots(figsize=(8, 5.4)); p(ax); f.tight_layout()
    f.savefig(OUT / f"S13_why_{i}.png", bbox_inches="tight", pad_inches=0.12); plt.close(f)
f, axs = plt.subplots(2, 2, figsize=(16, 11))
for p, ax in zip(panels, axs.ravel()): p(ax)
f.tight_layout(h_pad=2.5, w_pad=2.5)
f.savefig(OUT / "S13_why_grid.png", bbox_inches="tight", pad_inches=0.15); plt.close(f)
print("ok", OUT)
