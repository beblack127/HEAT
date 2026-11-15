"""결과 리포트(00_결과_리포트.md)용 그림.

08_fig_market.png       서울 연도별 개업·폐업, 월별 영업 점포 수
08_fig_season.png       서울 달별 계절 계수 (개업·폐업)
08_fig_horizon.png      예측 거리별 MASE: 가장 좋은 모델 vs 계절 naive
08_fig_levels.png       집계 수준별 MASE (개업, h = 3)
08_fig_cross.png        교차 피처를 넣었을 때 오차 변화 (DM-HLN)
08_fig_persistence.png  동 순위 지속성, 검색 비중 변동·되돌림
08_fig_importance.png   LightGBM 피처 중요도 상위 8개
08_fig_hotspots.png     핫스팟·떠오르는 곳 4개 리스트
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C, log_run, read_csv, set_korean_font
import data as D

set_korean_font()
R = C.REPORTS
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
FOLD_C = {"F2024": BLUE, "F2025": ORANGE}
plt.rcParams.update({"axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "font.size": 9, "legend.frameon": False, "figure.facecolor": "white"})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e5e0", lw=.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(R / name, dpi=150, facecolor="white")
    plt.close(fig)


met = pd.read_csv(R / "06_metrics.csv")
dm = pd.read_csv(R / "06_dm.csv")

# ---------------- 1. 시장 추세 ----------------
O = D.wide("open", "seoul").iloc[:, 0]
Cd = D.wide("close_v_drop", "seoul").iloc[:, 0]
S = D.stock("seoul").iloc[:, 0]
yrs = ["2022", "2023", "2024", "2025"]
oy = O.groupby(O.index.str[:4]).sum().loc[yrs]
cy = Cd.groupby(Cd.index.str[:4]).sum().loc[yrs]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
x = np.arange(len(yrs))
for i, (v, c, lb) in enumerate(((oy, BLUE, "개업"), (cy, ORANGE, "폐업 (일괄 폐업 제외)"))):
    b = ax[0].bar(x + (i - .5) * .38, v.values / 1000, .36, color=c, label=lb)
    ax[0].bar_label(b, labels=[f"{t / 1000:.1f}" for t in v.values], fontsize=8, color=INK2, padding=2)
ax[0].set_xticks(x, yrs)
ax[0].set_ylabel("건 (천)")
ax[0].set_ylim(0, 24)
ax[0].set_title("서울 연도별 개업·폐업")
ax[0].legend(loc="upper left", ncol=2)
grid(ax[0])
s = S.loc["2022-01":"2025-12"] / 1000
ax[1].plot(s.index, s.values, color=BLUE, lw=2)
pk = s.idxmax()
ax[1].scatter([pk], [s[pk]], s=40, color=BLUE, edgecolor="white", lw=2, zorder=3)
ax[1].annotate(f"최고 {s[pk]:.1f}천 ({pk})", (pk, s[pk]), xytext=(10, -4), textcoords="offset points", fontsize=8, color=INK2)
ax[1].annotate(f"{s.iloc[-1]:.1f}천 (2025-12)", (s.index[-1], s.iloc[-1]), xytext=(-95, 4), textcoords="offset points",
               fontsize=8, color=INK2)
t = [m for m in s.index if m.endswith("-01")]
ax[1].set_xticks(t, [m[:4] for m in t])
ax[1].set_ylabel("영업 점포 (천)")
ax[1].set_title("서울 영업 점포 수 (월말)")
grid(ax[1])
save(fig, "08_fig_market.png")

# ---------------- 2. 계절성 ----------------
fig, ax = plt.subplots(figsize=(10, 3.8))
mo = [f"{m:02d}" for m in range(1, 13)]
for i, (y, c, lb) in enumerate(((O, BLUE, "개업"), (Cd, ORANGE, "폐업 (일괄 폐업 제외)"))):
    y = y.loc["2022-01":"2025-12"]
    r = (y / y.rolling(12, center=True).mean()).groupby(y.index.str[5:]).mean().reindex(mo)
    ax.bar(np.arange(12) + (i - .5) * .38, r.values - 1, .36, bottom=1, color=c, label=lb)
ax.axhline(1, color=INK2, lw=.8)
ax.set_xticks(range(12), [f"{m}월" for m in range(1, 13)])
ax.set_ylim(.75, 1.25)
ax.set_ylabel("계절 계수 (1 = 연평균)")
ax.set_title("서울 개업·폐업의 달별 계절 계수 (2022~2025)")
ax.legend(loc="upper left", ncol=2)
grid(ax)
save(fig, "08_fig_season.png")

# ---------------- 3. 예측 거리 ----------------
TG = [("open", "개업 수"), ("close_v_drop", "폐업 수"), ("net_rate_v_drop", "순증률"), ("search", "검색 비중")]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), sharey=False)
for ax, (t, ti) in zip(axs, TG):
    m = met[(met.target == t) & (met.level == "dong") & (met.scope == "all")]
    for f in ("F2024", "F2025"):
        mm = m[m.fold == f]
        best = mm.groupby("h").MASE_mean.min()
        sn = mm[mm.model == "SeasonalNaive"].set_index("h").MASE_mean
        if t == "search":   # h = 12 F2024는 분모가 origin 1개라 제외
            best, sn = best.drop(12, errors="ignore") if f == "F2024" else best, \
                sn.drop(12, errors="ignore") if f == "F2024" else sn
        ax.plot(best.index, best.values, "-o", color=FOLD_C[f], lw=2, ms=6, mec="white", mew=1.5, label=f"{f} 가장 좋은 모델")
        ax.plot(sn.index, sn.values, "--o", color=FOLD_C[f], lw=1.2, ms=5, mfc="white", label=f"{f} 계절 naive")
    ax.axhline(1, color=INK2, lw=.8, ls=":")
    ax.set_xticks([3, 6, 12], ["3개월", "6개월", "12개월"])
    ax.set_title(ti)
    grid(ax)
axs[0].set_ylabel("MASE (1 = 학습 구간 계절 naive)")
axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle("예측 거리가 길수록 기준선과의 차이가 사라진다 (동 단위)", fontsize=11, fontweight="bold", color=INK, x=.01, ha="left")
save(fig, "08_fig_horizon.png")

# ---------------- 4. 집계 수준 ----------------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
LV = [("seoul", "서울"), ("gu", "구 (25)"), ("dong", "동 (427)")]
for ax, f in zip(axs, ("F2024", "F2025")):
    m = met[(met.target == "open") & (met.h == 3) & (met.scope == "all") & (met.fold == f)]
    best = [m[m.level == lv].MASE_mean.min() for lv, _ in LV]
    bname = [m[m.level == lv].sort_values("MASE_mean").model.iloc[0] for lv, _ in LV]
    sn = [m[(m.level == lv) & (m.model == "SeasonalNaive")].MASE_mean.iloc[0] for lv, _ in LV]
    x = np.arange(3)
    b1 = ax.bar(x - .19, best, .36, color=BLUE, label="가장 좋은 모델")
    b2 = ax.bar(x + .19, sn, .36, color=GRAY, label="계절 naive")
    ax.bar_label(b1, labels=[f"{v:.2f}" for v in best], fontsize=8, color=INK2, padding=2)
    ax.bar_label(b2, labels=[f"{v:.2f}" for v in sn], fontsize=7, color=INK2, padding=2)
    ax.axhline(1, color=INK2, lw=.8, ls=":")
    ax.set_xticks(x, [f"{n}\n가장 좋은 모델: {bn}" for (_, n), bn in zip(LV, bname)], fontsize=8)
    ax.set_title(f"개업 수 MASE, h = 3 · {f}")
    ax.set_ylim(0, 1.55)
    grid(ax)
axs[0].set_ylabel("MASE")
axs[0].legend(loc="upper left")
save(fig, "08_fig_levels.png")

# ---------------- 5. 교차 피처 ----------------
rows = []
for t, ti in [("search", "검색"), ("open", "개업"), ("close_v_drop", "폐업"), ("net_rate_v_drop", "순증률")]:
    for mod, base in (("LightGBM+x", "LightGBM"), ("Ridge+x", "Ridge")):
        for f in ("F2024", "F2025"):
            r = dm[(dm.target == t) & (dm.level == "dong") & (dm.scope == "all") & (dm.h == 3) & (dm.fold == f)
                   & (dm.model == mod) & (dm.vs == base)]
            if len(r):
                rows.append((f"{ti} · {base}", f, r.mean_loss_diff.iloc[0], r.p.iloc[0]))
cx = pd.DataFrame(rows, columns=["label", "fold", "diff", "p"])
labels = list(dict.fromkeys(cx.label))
fig, ax = plt.subplots(figsize=(9, 4.4))
y = np.arange(len(labels))
for i, f in enumerate(("F2024", "F2025")):
    d = cx[cx.fold == f].set_index("label").reindex(labels)
    b = ax.barh(y + (i - .5) * .38, d["diff"].values, .36, color=FOLD_C[f], label=f)
    ax.bar_label(b, labels=[("**" if p < .05 else ("*" if p < .1 else "")) for p in d.p.values], fontsize=9, color=INK, padding=2)
ax.axvline(0, color=INK2, lw=.8)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("교차 피처를 넣었을 때 척도 오차 변화 (오른쪽 = 나빠짐, 왼쪽 = 좋아짐)")
ax.set_title("검색 ↔ 개폐업 교차 피처의 효과 (h = 3, 동) · ** p<0.05, * p<0.1")
ax.legend(loc="upper right")
grid(ax, "x")
save(fig, "08_fig_cross.png")

# ---------------- 6. 지속성·되돌림 ----------------
sh = D.search_wide("share")
On = D.wide("open", "dong")
Cdn = D.wide("close_v_drop", "dong")
Sn = D.stock("dong")
ann_s = sh.groupby(sh.index.str[:4]).mean()
ann_o = On.loc["2022":].groupby(On.loc["2022":].index.str[:4]).sum()
ann_c = Cdn.loc["2022":].groupby(Cdn.loc["2022":].index.str[:4]).sum()
ann_n = (ann_o - ann_c) / Sn.loc[[f"{y}-01" for y in ann_o.index]].set_axis(ann_o.index)
pairs = [("2022", "2023"), ("2023", "2024"), ("2024", "2025")]
rc = {lb: [a.loc[p].corr(a.loc[q], method="spearman") for p, q in pairs]
      for lb, a in (("검색 비중", ann_s), ("개업 수", ann_o), ("순증률", ann_n))}
ma = sh.rolling(3).mean()
past, fut = np.log(ma / ma.shift(3)), np.log(ma.shift(-3) / ma)
mr, vol = [], []
for yr in ("2023", "2024", "2025"):
    idx = [i for i in ma.index if i.startswith(yr) and fut.loc[i].notna().sum() > 100]
    mr.append(np.nanmean([past.loc[i].corr(fut.loc[i], method="spearman") for i in idx]))
    idx2 = [i for i in sh.index if i.startswith(yr)]
    vol.append(np.log(sh / sh.shift(1)).loc[idx2].abs().median().median() * 100)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
x = np.arange(3)
for i, (lb, c) in enumerate(zip(rc, (BLUE, ORANGE, AQUA))):
    b = ax[0].bar(x + (i - 1) * .27, rc[lb], .25, color=c, label=lb)
    ax[0].bar_label(b, labels=[f"{v:.2f}" for v in rc[lb]], fontsize=7, color=INK2, padding=2)
ax[0].set_xticks(x, [f"{p}→{q}" for p, q in pairs])
ax[0].set_ylim(0, 1.12)
ax[0].set_ylabel("동 순위상관 (스피어만)")
ax[0].set_title("해마다 동 순위가 유지되는 정도")
ax[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, -.1))
b = ax[1].bar(["2023", "2024", "2025"], mr, .5, color=BLUE)
ax[1].bar_label(b, labels=[f"{v:.2f}" for v in mr], fontsize=8, color=INK2, padding=2)
ax[1].axhline(0, color=INK2, lw=.8)
ax[1].set_ylim(-.45, .05)
ax[1].set_title("검색 비중: 최근 3개월 변화 vs\n다음 3개월 변화 (순위상관)")
b = ax[2].bar(["2023", "2024", "2025"], vol, .5, color=ORANGE)
ax[2].bar_label(b, labels=[f"{v:.1f}%" for v in vol], fontsize=8, color=INK2, padding=2)
ax[2].set_ylim(0, 8)
ax[2].set_title("검색 비중: 동별 월 변동폭\n(중앙값)")
for a in ax:
    grid(a)
save(fig, "08_fig_persistence.png")

# ---------------- 7. 피처 중요도 ----------------
imp = pd.read_csv(R / "05_feature_importance.csv")
NICE = {"srch_sf_ratio": "앞 3개월 계절계수 ÷ 최근", "own_s_ma12": "검색 12개월 평균(상대)", "own_s_l0": "이번 달 검색(상대)",
        "srch_chg_3m": "검색 3개월 변화", "srch_chg_3m_sa": "검색 3개월 변화(계절 보정)", "cal_month": "달",
        "gu_cat": "구", "own_s_ma6": "검색 6개월 평균(상대)", "cal_dong_level": "동 평균 수준(학습 구간)",
        "own_open_ma12": "개업 12개월 평균", "own_close_ma12": "폐업 12개월 평균", "fnb_logS": "영업 점포 수(log)",
        "own_open_ma6": "개업 6개월 평균", "own_close_ma6": "폐업 6개월 평균", "own_open_snaive": "1년 전 같은 기간 개업",
        "fnb_age_mean": "영업 점포 평균 업력", "fnb_net_12m": "12개월 순증률", "fnb_trend_d12": "트렌드 업종 비중 1년 변화",
        "fnb_cafe_share": "카페 비중", "fnb_S_g12": "점포 수 1년 증가율", "srch_chg_12m": "검색 1년 변화",
        "srch_log_ma3": "검색 비중 수준", "srch_level_pct": "검색 수준 백분위",
        "cal_covid_fut": "코로나 더미(예측 구간)", "srch_chg_1m": "검색 1개월 변화"}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (t, ti) in zip(axs, TG):
    x = imp[(imp.target == t) & (imp.h == 3) & (imp.group == "base")]
    g = x.groupby("feature").gain.sum()
    g = (g / g.sum()).sort_values(ascending=False).head(8)[::-1]
    b = ax.barh([NICE.get(k, k) for k in g.index], g.values * 100, color=BLUE, height=.6)
    ax.bar_label(b, labels=[f"{v * 100:.0f}%" for v in g.values], fontsize=7, color=INK2, padding=2)
    ax.set_title(ti)
    ax.set_xlim(0, max(g.values) * 125)
    ax.tick_params(axis="y", labelsize=8)
    grid(ax, "x")
fig.suptitle("LightGBM 피처 중요도 상위 8개 (gain 비중, h = 3, 두 fold 합)", fontsize=11, fontweight="bold",
             color=INK, x=.01, ha="left")
save(fig, "08_fig_importance.png")

# ---------------- 8. 핫스팟 4개 리스트 ----------------
H = {k: read_csv(R / f"07_hotspots_{k}.csv") for k in ("search_hot", "search_emerging", "fnb_hot", "fnb_emerging")}
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
lab = lambda d: [f"{i}. {n}" for i, n in zip(d["순위"], d["동"])][::-1]  # noqa: E731

d = H["search_hot"]
ax = axs[0, 0]
b = ax.barh(lab(d), d["서울대비비중_%"][::-1], color=BLUE, height=.6)
ax.bar_label(b, labels=[f"{v:.2f}%  (1년 {c + 0:+.0f}%)".replace("+0%", "0%").replace("-0%", "0%") for v, c in zip(d["서울대비비중_%"][::-1], d["1년전대비_%"][::-1])],
             fontsize=8, color=INK2, padding=3)
ax.set_xlim(0, d["서울대비비중_%"].max() * 1.45)
ax.set_title("검색 · 현재 핫스팟: 서울 대비 검색 비중 (2026-08~10)")

d = H["search_emerging"]
ax = axs[0, 1]
y = np.arange(len(d))
ax.barh(y + .19, d["1년전대비_%"][::-1], .36, color=BLUE, label="1년 전 같은 3개월 대비")
ax.barh(y - .19, d["6개월전대비_%"][::-1], .36, color=ORANGE, label="6개월 전 3개월 대비")
for yi, v in zip(y, d["1년전대비_%"][::-1]):
    ax.text(v + 1, yi + .19, f"+{v:.0f}%", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, lab(d))
ax.set_xlim(0, d["1년전대비_%"].max() * 1.2)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("검색 · 떠오르는 곳: 검색 비중 증가율 (%)")

d = H["fnb_hot"]
ax = axs[1, 0]
y = np.arange(len(d))
ax.barh(y + .19, d["개업_12개월"][::-1], .36, color=BLUE, label="개업")
ax.barh(y - .19, d["폐업_12개월"][::-1], .36, color=ORANGE, label="폐업 (일괄 폐업 포함)")
for yi, v in zip(y, d["개업_12개월"][::-1]):
    ax.text(v + 4, yi + .19, f"{v}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, lab(d))
ax.legend(loc="lower right", fontsize=8)
ax.set_title("개폐업 · 현재 핫스팟: 12개월 개업 수 (2025-10~2026-09)")

d = H["fnb_emerging"]
ax = axs[1, 1]
y = np.arange(len(d))
ax.barh(y + .19, d["점포증가율_%"][::-1], .36, color=BLUE, label="최근 1년 (2025-09→2026-09)")
ax.barh(y - .19, d["전년증가율_%"][::-1], .36, color=GRAY, label="그 전 1년 (2024-09→2025-09)")
ax.set_xlim(-17, 16)
for yi, v in zip(y, d["점포증가율_%"][::-1]):
    ax.text(v + .3, yi + .19, f"+{v:.1f}%", va="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=.8)
ax.set_yticks(y, lab(d))
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2, fontsize=8)
ax.set_title("개폐업 · 떠오르는 곳: 영업 점포 증가율 (%)")
for a in axs.flat:
    grid(a, "x")
    a.tick_params(axis="y", labelsize=9)
save(fig, "08_fig_hotspots.png")

log_run("08_report_figures", "리포트 그림 8개")
print("완료")
