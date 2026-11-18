"""11쪽 'Why extra data didn't help' 오른쪽 그림: 붐빔 vs 상승 / 입력 추가 / 가짜 입력.
출력: reports/gt_midterm/slides/fig/S14_whynot_{1..3}.png, S14_whynot_grid.png
"""
import glob, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import config as C

OUT = C.REPORT / "slides" / "fig"; TB = C.TABLES
for f in glob.glob(str(pathlib.Path.home() / ".fonts/KaiseiTokumin-*.ttf")): fm.fontManager.addfont(f)
BG, NAVY, RED, INK, GREY, LIGHT = "#F7F7F7", "#004490", "#AE3A1E", "#222222", "#8A8F98", "#C9CED6"
plt.rcParams.update({"font.family": "Kaisei Tokumin", "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 18, "xtick.labelsize": 17, "ytick.labelsize": 17,
                     "legend.fontsize": 16, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "savefig.dpi": 200})

v1 = pd.read_csv(TB / "v1_correlation.csv").set_index("자료")
v2 = pd.read_csv(TB / "v2_change_cross.csv"); v2 = v2[v2["기간(1~7월)"] == "2023→2025"].set_index("자료")
t9 = pd.read_csv(TB / "t9_clean_inputs.csv", index_col=[0, 1, 2])
pl = pd.read_csv(TB / "t10_placebo.csv", index_col=[0, 1])


def p1(ax):
    """붐비는 곳(수준) vs 늘어나는 곳(변화): 동 순위 상관"""
    items = [("F&B stores", "음식점 수"), ("Subway exits", "지하철 하차"), ("Search share", "검색 비중")]
    y = np.arange(len(items))
    for yi, (lab, k) in zip(y, items):
        a, b = v1.loc[k, "스피어만 상관"], v2.loc[k, "변화 순위 상관"]
        ax.plot([b, a], [yi, yi], color=LIGHT, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [yi], s=260, color=NAVY, zorder=3); ax.scatter([b], [yi], s=260, color=RED, zorder=3)
        ax.text(a, yi + .28, f"{a:.2f}", ha="center", fontsize=17, color=NAVY, fontweight="bold")
        ax.text(b, yi + .28, f"{b:.2f}", ha="center", fontsize=17, color=RED, fontweight="bold")
    ax.set_yticks(y, [i[0] for i in items]); ax.set_ylim(-.6, len(items) - .2); ax.set_xlim(0, 1)
    ax.set_xlabel("Rank correlation with mobility (all dongs)")
    ax.scatter([], [], s=160, color=NAVY, label="Where it is busy (level)")
    ax.scatter([], [], s=160, color=RED, label="Where it is rising (change, 2023→25)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, handletextpad=.3)
    ax.set_title("Busy ≠ rising", loc="left", color=NAVY, fontweight="bold", pad=44)


def bars(ax, labs, vals, cols, ylim, ttl, note, d=1):
    x = np.arange(len(vals))
    r = ax.bar(x, vals, .6, color=cols)
    for xi, v in zip(x, vals): ax.text(xi, v + (ylim[1] - ylim[0]) * .02, f"{v:.{d}f}%", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, labs); ax.set_ylim(*ylim); ax.set_ylabel("WAPE (%), 4 dates")
    ax.set_title(ttl, loc="left", color=NAVY, fontweight="bold")
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=15, color=GREY)


def p2(ax):
    """대표 모델에 입력 추가"""
    k = [("None", "입력 없음"), ("+ Subway", "지하철"), ("+ F&B", "개폐업"), ("+ Search", "검색"), ("+ All 3", "셋 다")]
    v = [t9.loc[(1, "ARIMA+푸리에 (보정)", i[1]), "4시점"] for i in k]
    bars(ax, [i[0] for i in k], v, [RED] + [LIGHT] * 4, (10, 12.6), "Adding inputs", "on our best model", d=2)


def p3(ax):
    """가짜 입력: 다른 동의 검색량"""
    k = [("None", "none"), ("Own search", "search_index"), ("Random dong's\nsearch", "placebo")]
    v = [pl.loc[("s13", i[1]), "4시점"] for i in k]
    bars(ax, [i[0] for i in k], v, [GREY, NAVY, LIGHT], (20, 25), "Placebo test", "first-draft setup")


panels = [p1, p2, p3]
for i, p in enumerate(panels, 1):
    f, ax = plt.subplots(figsize=(8, 5.4) if i > 1 else (16, 5)); p(ax); f.tight_layout()
    f.savefig(OUT / f"S14_whynot_{i}.png", bbox_inches="tight", pad_inches=0.12); plt.close(f)
f = plt.figure(figsize=(16, 11))
gs = f.add_gridspec(2, 2, height_ratios=[1, 1.15])
p1(f.add_subplot(gs[0, :])); p2(f.add_subplot(gs[1, 0])); p3(f.add_subplot(gs[1, 1]))
f.tight_layout(h_pad=2.5, w_pad=2.5)
f.savefig(OUT / "S14_whynot_grid.png", bbox_inches="tight", pad_inches=0.15); plt.close(f)
print("ok")

# 두 칸 (입력 추가 → 붐빔≠상승), 왼쪽 4칸 그림과 같은 크기
f = plt.figure(figsize=(16, 11))
gs = f.add_gridspec(2, 1, height_ratios=[1, 1.1])
p2(f.add_subplot(gs[0])); p1(f.add_subplot(gs[1]))
f.tight_layout(h_pad=3)
f.savefig(OUT / "S14_whynot_2panel.png", bbox_inches="tight", pad_inches=0.15); plt.close(f)
print("ok 2panel")
