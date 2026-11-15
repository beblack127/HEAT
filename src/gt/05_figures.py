"""그림 (명세 §8). 예측은 주 단위, 그림은 13주 예측 구간 중심. reports/gt_midterm/fig/*.png"""
import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from common import set_korean_font, unit_names, units_of_dongs

set_korean_font()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
BLUE, GREY, RED, GREEN, INK = "#2F5FB3", "#8A8F98", "#D4462A", "#2F8F5B", "#1C2430"
NM = unit_names(); BYNAME = {v[1]: k for k, v in NM.items()}
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C.RESULTS / "pred_*.parquet")))], ignore_index=True)
P["key"] = P.model + "|" + P.xset
W = pd.read_parquet(C.RESULTS / "wape_by_origin.parquet")
U = pd.read_parquet(C.RESULTS / "unit_metrics.parquet")
t1 = pd.read_csv(C.TABLES / "t1_main_wape.csv", index_col=0, encoding="utf-8-sig")
K = pd.read_csv(C.TABLES / "topk.csv", encoding="utf-8-sig")
H = pd.read_csv(C.TABLES / "hotspots.csv", encoding="utf-8-sig", dtype={"unit": str})
MAIN_O = pd.Timestamp("2025-04-27")
LAB = {"naive": "Naive", "snaive": "작년 같은 주", "arima": "ARIMA", "sarima": "SARIMA", "arimax": "ARIMAX", "sarimax": "SARIMAX"}
XLAB = {"none": "입력 없음", "search": "검색", "all_noLP": "전체−생활인구", "all": "전체", "search_idx": "검색(원지수)"}
save = lambda f, n: (f.savefig(C.FIG / n, bbox_inches="tight"), plt.close(f))


def uwape(u, o, key):
    g = P[(P.unit == u) & (P.origin == o) & (P.key == key)].merge(T[["unit", "week_end", "y"]], on=["unit", "week_end"])
    return (g.y - g.y_hat).abs().sum() / g.y.sum() * 100 if len(g) else np.nan


def case_panel(ax, u, o, a_key, b_key, title, zoom):
    s = T[T.unit == u].set_index("week_end").y
    o = pd.Timestamp(o); end = o + pd.Timedelta(weeks=C.H)
    if zoom: s = s[(s.index > o - pd.Timedelta(weeks=20)) & (s.index <= end)]
    else: s = s[s.index <= end]
    ax.plot(s[s.index <= o], color=BLUE, lw=1.6, label="예측 시점까지 실제")
    ax.plot(s[s.index >= o], color=BLUE, lw=1.4, ls="--", label="이후 실제")
    ax.axvspan(o, end, color=BLUE, alpha=.06, lw=0); ax.axvline(o, color=INK, lw=.8, ls=":")
    for key, col, lw, ls, lab in ((("snaive|none"), GREEN, 1.1, ":", "작년 같은 주"), (a_key, GREY, 1.6, "-", "입력 없음"), (b_key, RED, 1.8, "-", "입력 있음")):
        g = P[(P.unit == u) & (P.origin == o) & (P.key == key)].sort_values("week_end")
        if len(g): ax.plot(g.week_end, g.y_hat, color=col, lw=lw, ls=ls, label=f"{lab}: {uwape(u, o, key):.1f}%")
    ax.set_title(title, loc="left"); ax.legend(frameon=False, loc="upper left"); ax.set_ylabel("일평균 도착 이동")
    ax.tick_params(axis="x", rotation=30)


def fig_case(name, zoom):
    u = BYNAME.get(name)
    if u is None or not len(P[P.unit == u]): return
    f, axs = plt.subplots(2, 2, figsize=(12, 7.2), sharey=False)
    specs = [("arima|none", "arimax|search", "ARIMA ± 검색"), ("sarima|none", "sarimax|search", "SARIMA ± 검색"),
             ("arima|none", "arimax|all", "ARIMA ± 전체 입력"), ("sarima|none", "sarimax|all", "SARIMA ± 전체 입력")]
    for ax, (a, b, t) in zip(axs.flat, specs): case_panel(ax, u, MAIN_O, a, b, t, zoom)
    best = min(["arima|none", "sarima|none", "arimax|search", "sarimax|search", "arimax|all", "sarimax|all", "snaive|none"], key=lambda k: uwape(u, MAIN_O, k))
    f.suptitle(f"{name}: 13주 오차가 가장 작은 것은 {LAB[best.split('|')[0]]}{'' if best.endswith('none') else ' + ' + XLAB[best.split('|')[1]]} ({uwape(u, MAIN_O, best):.1f}%)  ·  예측 시점 2025-04-27", x=0.01, ha="left", fontsize=12)
    f.tight_layout(); save(f, f"F{2 if zoom else 1}_case_{name}.png")


# 사례 동 고르기 (규칙): 대표 모델(snaive 제외 평균 WAPE 최저)의 2025-04-27 동별 WAPE로
#  잘 맞춘 곳 = 규모(직전 13주 평균) 중앙값 이상 동 중 WAPE 최저, 보통 = 전체 중앙값, 못 맞춘 곳 = 상위 10% 경계
REP = W.groupby("key").wape.mean().drop(labels=["snaive|none"], errors="ignore").idxmin()
ur = U[(U.origin == MAIN_O) & (U.key == REP)].set_index("unit").wape.dropna()
lvl = T[(T.week_end <= MAIN_O) & (T.week_end > MAIN_O - pd.Timedelta(weeks=C.H))].groupby("unit").y.mean()
big = ur[ur.index.isin(lvl[lvl >= lvl.median()].index)]
pick = {"잘 맞춘 곳": big.idxmin(),
        "보통인 곳": (ur - ur.median()).abs().idxmin(),
        "못 맞춘 곳": (ur - ur.quantile(.9)).abs().idxmin()}
CASES = []
for role, u in pick.items():
    nm = NM.get(u, ("", u))[1]; rk = int(ur.rank().loc[u]); CASES.append((role, u, nm, ur.loc[u], rk, len(ur)))
    BYNAME[nm] = u
    fig_case(nm, False); fig_case(nm, True)
pd.DataFrame(CASES, columns=["role", "unit", "dong", "wape_rep", "rank_low_is_good", "n"]).assign(rep_model=REP).to_csv(C.TABLES / "case_dongs.csv", index=False, encoding="utf-8-sig")
print("사례:", CASES)

# F3 히트맵
f, ax = plt.subplots(figsize=(7.5, 4.2))
M = t1.values.astype(float)
im = ax.imshow(np.where(np.isnan(M), np.nan, M), cmap="Blues_r", aspect="auto")
ax.set_xticks(range(M.shape[1]), [XLAB.get(c, c) for c in t1.columns]); ax.set_yticks(range(M.shape[0]), [LAB.get(r, r) for r in t1.index])
best = np.nanmin(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:.1f}%", ha="center", va="center", fontsize=9, color="white" if M[i, j] < np.nanpercentile(M, 35) else INK, fontweight="bold" if M[i, j] == best else "normal")
ax.set_title(f"4개 시점 평균 13주 WAPE (낮을수록 좋음). 가장 좋은 칸 {best:.1f}%", loc="left")
for s in ax.spines.values(): s.set_visible(False)
save(f, "F3_heatmap_wape.png")

# F4 시점별 막대
keys = ["naive|none", "arima|none", "sarima|none", "arimax|all", "sarimax|all", "arimax|all+py"]
pv = W.pivot(index="origin", columns="key", values="wape")
f, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(pv)); wdt = 0.13
cols = ["#C9CED6", "#9AA6B8", "#6E86A8", "#E8A07F", "#D4462A", "#7A2513"]
for i, k in enumerate(keys):
    if k in pv: ax.bar(x + (i - len(keys) / 2) * wdt + wdt / 2, pv[k], wdt, color=cols[i], label=f"{LAB[k.split('|')[0]]}{'' if k.endswith('none') else ' + ' + XLAB.get(k.split('|')[1].replace('+py', ''), '') + (' + 작년 같은 주' if '+py' in k else '')}")
for i, o in enumerate(pv.index):
    ax.hlines(pv.loc[o, "snaive|none"], i - .45, i + .45, color=GREEN, lw=2)
ax.plot([], [], color=GREEN, lw=2, label="작년 같은 주 (기준선)")
ax.set_xticks(x, [str(pd.Timestamp(o).date()) for o in pv.index]); ax.set_ylabel("13주 WAPE (%)")
ax.set_title("예측 시점별 13주 오차: 초록 선(작년 같은 주)보다 낮으면 기준선보다 나음", loc="left"); ax.legend(ncol=4, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.12))
save(f, "F4_by_origin.png")

# F5 동별 분포
ks = ["snaive|none", "naive|none", "arima|none", "sarima|none", "arimax|all", "sarimax|all"]
f, ax = plt.subplots(figsize=(9, 3.8))
ax.boxplot([U[U.key == k].wape.clip(lower=0.1) for k in ks], tick_labels=[LAB[k.split("|")[0]] + ("" if k.endswith("none") else "+전체") for k in ks], showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#E3E9F3", color=GREY), medianprops=dict(color=RED, lw=1.6))
ax.set_yscale("log"); ax.set_ylabel("동별 13주 WAPE (%), 로그 축"); ax.set_title("동마다 오차가 크게 다르다: 상자 = 가운데 50% 동, 빨간 선 = 중앙값", loc="left")
save(f, "F5_dong_dist.png")

# F6 M3
t3 = pd.read_csv(C.TABLES / "t3_prior_year.csv", encoding="utf-8-sig")
f, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(t3))
ax.bar(x - .18, t3.without_prior_year, .36, color=GREY, label="작년 같은 주 입력 없음")
ax.bar(x + .18, t3.with_prior_year, .36, color=RED, label="작년 같은 주 입력 있음")
ax.axhline(t3.snaive.iloc[0], color=GREEN, lw=2, label="작년 같은 주 (기준선)")
ax.set_xticks(x, ["ARIMA" if s == "none" else f"ARIMAX + {XLAB[s]}" for s in t3.xset]); ax.set_ylabel("평균 13주 WAPE (%)")
ax.set_title("M3: 작년 같은 주 값을 입력으로 넣었을 때", loc="left"); ax.legend(frameon=False)
save(f, "F6_prior_year.png")

# F7 핫스팟 지도
try:
    import geopandas as gpd
    g = gpd.read_parquet(C.EMD)
    g = g[g.sidocd.astype(str) == "11"].copy()
    g["dong_code8"] = g.emdcd.astype(str).str[:8]
    g["unit"] = g.dong_code8.map(units_of_dongs())
    h = H[(H.origin == str(MAIN_O.date())) & (H.window == "now")]
    hot, em = set(h[h.list == "hot"].unit), set(h[h.list == "emerging"].unit)
    f, ax = plt.subplots(figsize=(8.5, 7))
    g.plot(ax=ax, color=["#D4462A" if u in hot else "#F6E3DA" for u in g.unit], edgecolor="white", lw=.3)
    g[g.unit.isin(em)].boundary.plot(ax=ax, color="#1C2430", lw=1.6)
    ax.set_axis_off()
    ax.set_title(f"2025-04-27 직전 13주: 현재 핫스팟 {len(hot)}곳(빨강), 떠오르는 곳 {len(em)}곳(검은 테두리)", loc="left")
    save(f, "F7_hotspot_map.png")
except Exception as e:
    print("F7 생략:", e)

# F8 상위 K개 적중률
k20 = K[K.K.astype(str).isin(["10", "20", "30"])].copy(); k20["K"] = k20.K.astype(int)
best_keys = W.groupby("key").wape.mean().drop(labels=["snaive|none"], errors="ignore").nsmallest(2).index.tolist()
f, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, tgt, ttl in ((axs[0], "emerging", "떠오르는 곳"), (axs[1], "hot", "현재 핫스팟")):
    d = k20[k20.target == tgt]; Ks = [10, 20, 30]; x = np.arange(3)
    for i, (m, col) in enumerate([(best_keys[0], RED), (best_keys[1], "#E8A07F"), ("persistence", BLUE)]):
        v = d[d.method == m].groupby("K").hit_rate.mean().reindex(Ks)
        lab = "지속(직전 13주 그대로)" if m == "persistence" else f"{LAB[m.split('|')[0]]}{'' if m.endswith('none') else ' + ' + XLAB.get(m.split('|')[1].replace('+py', ''), m.split('|')[1])}"
        ax.bar(x + (i - 1) * .26, v, .26, color=col, label=lab)
    rnd = d[d.method == "random"].groupby("K").hit_rate.mean().reindex(Ks)
    for i, r in enumerate(rnd): ax.hlines(r, i - .4, i + .4, color=INK, lw=1.5, ls="--")
    ax.plot([], [], color=INK, ls="--", label="무작위")
    ax.set_xticks(x, [f"상위 {k}" for k in Ks]); ax.set_title(f"{ttl}: 예측 목록이 실제와 겹친 비율 (4개 시점 평균)", loc="left"); ax.set_ylim(0, 1)
axs[0].set_ylabel("적중률"); axs[0].legend(frameon=False, fontsize=7.5)
save(f, "F8_topk.png")

# F10 서울 합계
s = T.groupby("week_end").seoul_y.first()
f, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(s, color=INK, lw=1.4)
for a, b in (("2024-08-01", "2024-10-31"), ("2025-09-01", str(s.index.max().date()))):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#F2A47E", alpha=.25, lw=0)
ax.set_ylabel("서울 합계 (일평균)"); ax.set_title("서울 전체 20~39세 목적 4+5+7 도착 이동 (주): 주황 = 품질 점검 구간. 전체가 함께 움직여서 떠오르는 곳은 서울 대비 비중으로 본다", loc="left", fontsize=10)
save(f, "F10_seoul_total.png")
print("그림 저장:", sorted(p.name for p in C.FIG.glob("*.png")))
