"""중간발표 슬라이드용 영어 그림 (1차 발표 디자인: 배경 #F7F7F7, 남색 #004490, 빨강 #AE3A1E, Kaisei Tokumin)
출력: reports/gt_midterm/slides/fig/*.png
"""
import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import config as C

OUT = C.REPORT / "slides" / "fig"; OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(__import__("pathlib").Path.home() / ".fonts/KaiseiTokumin-*.ttf")): fm.fontManager.addfont(f)
BG, NAVY, RED, INK, GREY, LIGHT, YEL = "#F7F7F7", "#004490", "#AE3A1E", "#222222", "#8A8F98", "#C9CED6", "#FEEFC5"
plt.rcParams.update({"font.family": "Kaisei Tokumin", "font.size": 21, "axes.titlesize": 23, "axes.labelsize": 20, "xtick.labelsize": 19, "ytick.labelsize": 19,
                     "legend.fontsize": 18, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "figure.dpi": 100, "savefig.dpi": 200})
TB = C.TABLES
save = lambda f, n: (f.savefig(OUT / n, bbox_inches="tight", pad_inches=0.12), plt.close(f))

# ---- S1 자료 이상: 생활이동 vs 카드 매출 ----
T = pd.read_parquet(C.INTERIM / "target_weekly.parquet")
s = T.groupby("week_end").seoul_y.first()
s = s[s.index <= "2025-12-31"]
q = pd.read_csv(TB / "v3_seoul_quarterly.csv", index_col=0)
q = q[q.index <= 20254]
qd = pd.to_datetime([f"{str(i)[:4]}-{(int(str(i)[4]) - 1) * 3 + 2:02d}-15" for i in q.index])
base = s["2024-01-01":"2024-06-30"].mean()
f, ax = plt.subplots(figsize=(14, 6))
ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-31"), color="#FECCBE", alpha=.7, lw=0)
ax.plot(s.index, s / base, color=NAVY, lw=2, label="Mobility (weekly, Seoul total)")
ax.plot(qd, q.food_cnt, color=RED, lw=2.4, marker="o", ms=7, label="Card sales, F&B (quarterly)")
ax.annotate("Aug–Oct 2024: mobility ×1.52,\ncard sales flat → likely a data issue", xy=(pd.Timestamp("2024-10-20"), 1.5), xytext=(pd.Timestamp("2024-11-20"), 1.62),
            fontsize=19, color=RED, fontweight="bold", va="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.set_ylabel("Index (2024 H1 = 1)"); ax.set_ylim(0.75, 1.75)
ax.legend(frameon=False, loc="upper left")
save(f, "S1_anomaly.png")

# ---- S2 타당성 ----
v1 = pd.read_csv(TB / "v1_correlation.csv").set_index("자료")
v3 = pd.read_csv(TB / "v3_sales.csv")
v2 = pd.read_csv(TB / "v2_change_cross.csv")
lvl = [("Card sales (F&B)", v3[(v3["비교"].str.startswith("수준")) & (v3["자료"] == "음식업 매출 건수")]["스피어만 상관"].iloc[0], True),
       ("F&B stores", v1.loc["음식점 수", "스피어만 상관"], False), ("Subway exits", v1.loc["지하철 하차", "스피어만 상관"], False),
       ("Search share", v1.loc["검색 비중", "스피어만 상관"], False)]
c2 = v2[v2["기간(1~7월)"] == "2023→2025"].set_index("자료")
chg = [("Card sales (F&B)*", v3[(v3["비교"] == "변화 2024→2025 (주)") & (v3["자료"] == "음식업 매출 건수")]["스피어만 상관"].iloc[0], True),
       ("F&B stores", c2.loc["음식점 수", "변화 순위 상관"], False), ("Subway exits", c2.loc["지하철 하차", "변화 순위 상관"], False),
       ("Search share", c2.loc["검색 비중", "변화 순위 상관"], False)]
f, axs = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
for ax, data, ttl in ((axs[0], lvl, "Where is hot?  (level, 2025)"), (axs[1], chg, "Where is rising?  (change)")):
    y = np.arange(len(data))[::-1]
    ax.barh(y, [d[1] for d in data], color=[RED if d[2] else NAVY for d in data], height=.6)
    for yi, d in zip(y, data): ax.text(d[1] + .02, yi, f"{d[1]:.2f}", va="center", fontsize=20, fontweight="bold")
    ax.set_yticks(y, [d[0] for d in data]); ax.set_xlim(0, 1); ax.set_title(ttl, loc="left", color=NAVY, fontweight="bold")
    ax.set_xlabel("Rank correlation (all dongs)")
axs[1].text(0.99, -0.32, "* not a model input · change: 2023→25 (sales 2024→25)", transform=axs[1].transAxes, ha="right", fontsize=15, color=GREY)
f.tight_layout(); save(f, "S2_validity.png")

# ---- S3 결과 ----
t7 = pd.read_csv(TB / "t7_more_models.csv", index_col=0)
col4, col2 = "4시점 평균", "2시점 평균(이상 구간 제외)"
rows = [("Naive\n(last week)", "Naive", "raw"), ("Same week\nlast year", "작년 같은 주", "raw"), ("ARIMA", "ARIMA", "raw"), ("SARIMA\n(52 weeks)", "SARIMA", "raw"),
        ("Same week\nlast year", "작년 같은 주 (보정)", "clean"), ("ARIMA", "ARIMA (보정)", "clean"), ("ARIMA\n+ Fourier", "ARIMA + 푸리에 (보정)", "clean"),
        ("Our model\n(combined)", "결합: 작년 같은 주 (보정) + ARIMA + 푸리에 (보정)", "clean")]
t9 = pd.read_csv(TB / "t9_clean_inputs.csv", index_col=[0, 1, 2])
f, ax = plt.subplots(figsize=(14.5, 7.6))
vals = [t7.loc[r[1], col4] for r in rows] + [t9.loc[(1, "ARIMA+푸리에 (보정)", "검색"), "4시점"]]
rows = rows + [("Our model\n+ search", "", "clean")]
x = np.arange(len(rows))
cols = [LIGHT if r[2] == "raw" else NAVY for r in rows]; cols[-2] = RED; cols[-1] = "#C98A78"
ax.bar(x, vals, color=cols, width=.62)
for xi, v in zip(x, vals): ax.text(xi, v + .4, f"{v:.1f}%", ha="center", fontsize=19, fontweight="bold")
ax.axhline(t7.loc["작년 같은 주", col4], color=GREY, ls="--", lw=1.2)
ax.axvline(3.5, color="#555555", lw=1, ls=":")
ax.text(1.5, 33.5, "Raw data", ha="center", fontsize=20, color=GREY, fontweight="bold")
ax.text(6, 33.5, "After cleaning Aug–Oct 2024", ha="center", fontsize=20, color=NAVY, fontweight="bold")
ax.set_xticks(x, [r[0] for r in rows], rotation=0, fontsize=15)
ax.set_ylabel("13-week WAPE (%), 4 origins"); ax.set_ylim(0, 36)
save(f, "S3_results.png")

# ---- S4 입력 조합 ----
try:
    t6 = pd.read_csv(TB / "t6_input_combo.csv", index_col=0)
    tt = pd.read_csv(TB / "t6_input_combo_tests.csv")
    sets = [("none", "None"), ("search", "Search"), ("subway", "Subway"), ("fnb", "F&B net\nopenings"), ("search_subway", "Search\n+Subway"),
            ("search_fnb", "Search\n+F&B"), ("subway_fnb", "Subway\n+F&B"), ("search_subway_fnb", "All three")]
    keys = ["arima|none" if s == "none" else f"arimax|{s}" for s, _ in sets]
    keys = [k for k in keys if k in t6.index]
    lab = [l for (s, l), k in zip(sets, ["arima|none" if s == "none" else f"arimax|{s}" for s, _ in sets]) if k in t6.index]
    v = t6.loc[keys, col4].values
    f, ax = plt.subplots(figsize=(13, 6.8))
    cols = [NAVY] + [LIGHT] * (len(keys) - 1)
    ax.bar(np.arange(len(keys)), v, color=cols, width=.62)
    for i, (k, vv) in enumerate(zip(keys, v)):
        ax.text(i, vv + .15, f"{vv:.1f}%", ha="center", fontsize=19, fontweight="bold")
        if i:
            r = tt[(tt.model == k) & (tt.vs == "arima|none")]
            if len(r): ax.text(i, vv - .9, f"{r.improved_share.iloc[0] * 100:.0f}% dongs\nimproved", ha="center", va="top", fontsize=14, color=INK)
    ax.set_xticks(np.arange(len(keys)), lab, fontsize=17); ax.set_ylim(15, max(v) + 1.2)
    ax.set_ylabel("13-week WAPE (%), ARIMA(X)\n(axis starts at 15)")
    save(f, "S4_inputs.png")
except FileNotFoundError:
    print("S4 생략: t6 없음")

# ---- S5 동네 예시 ----
EX, EXN, O = "11710631", "Garak 1-dong (Songpa-gu)", pd.Timestamp("2025-04-27")
P = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.RESULTS / "pred_comb_*.parquet")) + glob.glob(str(C.RESULTS / "pred_more_*.parquet"))])
g = T[T.unit == EX].set_index("week_end").y
pr = P[(P.unit == EX) & (P.origin == O)].pivot_table(index="week_end", columns="model", values="y_hat")
win = g[(g.index > O - pd.Timedelta(weeks=26)) & (g.index <= O + pd.Timedelta(weeks=13))]
f, ax = plt.subplots(figsize=(12, 7))
ax.axvspan(O, O + pd.Timedelta(weeks=13), color="#E3E9F3", lw=0)
ax.plot(win[win.index <= O], color=NAVY, lw=2.4, label="Observed (known at decision)")
ax.plot(win[win.index >= O], color=NAVY, lw=2, ls="--", label="Observed (after)")
ax.plot(pr.index, pr.snaive_c, color=GREY, lw=1.6, ls=":", label="Same week last year")
ax.plot(pr.index, pr.comb_fourier, color=RED, lw=2.8, label="Our 13-week forecast")
yv = g.reindex(pr.index)
wr = (yv - pr.comb_fourier).abs().sum() / yv.sum() * 100; ws = (yv - pr.snaive_c).abs().sum() / yv.sum() * 100
ax.text(O + pd.Timedelta(days=4), ax.get_ylim()[1] * .99, f"Forecast error {wr:.1f}%\n(last year: {ws:.1f}%)", va="top", fontsize=19, color=RED, fontweight="bold")
ax.axvline(O, color=INK, lw=1, ls=":"); ax.text(O, ax.get_ylim()[0], " decision: 2025-04-27", fontsize=16, va="bottom")
ax.set_ylabel("Daily arrivals, age 20–39"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=16)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
save(f, "S5_example.png")
# ---- S6 사다리: 팀원 설정에서 한 가지씩 고치기 ----
try:
    LD = pd.read_csv(TB / "t8_ladder.csv")
    labs = ["First-draft\nsetup", "52-week\nseason", "Orders\nby AIC", "Search as\nSeoul share", "Cleaned\ndata", "+ same week\nlast year"]
    f, axs = plt.subplots(1, 2, figsize=(17, 6.6), sharey=True)
    for ax, (c1, c0, ttl) in zip(axs, (("검색O 04-27", "검색X 04-27", "Forecast date 2025-04-27 only"), ("검색O 2시점", "검색X 2시점", "Average of 2 dates without the anomaly"))):
        x = np.arange(len(LD)); w = .38
        ax.bar(x - w / 2, LD[c0], w, color=NAVY, label="Without search")
        ax.bar(x + w / 2, LD[c1], w, color=LIGHT, label="With search")
        for xi, a, b in zip(x, LD[c0], LD[c1]):
            ax.text(xi - w / 2, a + .5, f"{a:.1f}", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
            ax.text(xi + w / 2, b + .5, f"{b:.1f}", ha="center", fontsize=12.5, color=INK)
        ax.set_xticks(x, labs, fontsize=13.5); ax.set_title(ttl, loc="left", color=NAVY, fontweight="bold")
    axs[0].set_ylabel("13-week WAPE (%)"); axs[0].set_ylim(0, 41)
    axs[0].annotate("search helps only here", xy=(0.19, LD["검색O 04-27"][0] + 1.5), xytext=(-0.4, 25.5), fontsize=16, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axs[1].legend(frameon=False, loc="upper right")
    f.tight_layout(); save(f, "S6_ladder.png")
except FileNotFoundError:
    print("S6 생략: t8 없음")

# ---- S7 가짜 입력: 팀원 설정(13주 주기) ----
try:
    W10 = pd.read_csv(TB / "t10_placebo.csv", index_col=[0, 1])
    sets = [("none", "None"), ("search_index", "Search\n(own dong)"), ("placebo", "Search of a\nrandom dong"), ("subway_exit_4w", "Subway"), ("fnb_net3", "F&B net\nopenings")]
    f, axs = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, col, ttl in zip(axs, ("2025-04-27", "4시점"), ("Forecast date 2025-04-27 only", "Average of 4 forecast dates")):
        v = [W10.loc[("s13", s), col] for s, _ in sets]
        cl = [NAVY] + [LIGHT] * (len(sets) - 1); cl[2] = RED
        ax.bar(np.arange(len(sets)), v, color=cl, width=.62)
        for i, vv in enumerate(v): ax.text(i, vv + .1, f"{vv:.1f}%", ha="center", fontsize=17, fontweight="bold")
        lo = min(v); ax.set_ylim(lo - 3, max(v) + 1.2)
        ax.set_xticks(np.arange(len(sets)), [l for _, l in sets], fontsize=15); ax.set_title(ttl, loc="left", color=NAVY, fontweight="bold")
    axs[0].set_ylabel("13-week WAPE (%)\n(axis does not start at 0)")
    f.tight_layout(); save(f, "S7_placebo.png")
except FileNotFoundError:
    print("S7 생략: t10 없음")

print("저장:", sorted(p.name for p in OUT.glob("*.png")))

# ---- S8 사례 4곳 (2025-04-27): 대표 모델 최고·최악, 검색 추가로 가장 좋아진·나빠진 동 ----
try:
    cs = pd.read_csv(TB / "case4.csv", dtype={"unit": str}); jd = pd.read_csv(TB / "case4_judge.csv", dtype={"unit": str})
    S = pd.read_parquet(C.RESULTS / "case4_series.parquet")
    EN = {"11410640": "Hongje 3-dong (Seodaemun)", "11440630": "Sinsu-dong (Mapo)", "11320512": "Chang 2-dong (Dobong)", "11500604": "Gayang 2-dong (Gangseo)"}
    TT = {"rep_best": "Our model · best", "rep_worst": "Our model · worst", "search_best": "+ Search · helped most", "search_worst": "+ Search · hurt most"}
    O8 = pd.Timestamp("2025-04-27")
    f, axs = plt.subplots(2, 2, figsize=(17, 10.4))
    import matplotlib.dates as mdates
    for ax, (_, r) in zip(axs.flat, cs.iterrows()):
        u = r.unit; g = T[T.unit == u].set_index("week_end").y
        win = g[(g.index > O8 - pd.Timedelta(weeks=26)) & (g.index <= O8 + pd.Timedelta(weeks=13))]
        s = S[S.unit == u].set_index("week_end")
        ax.axvspan(O8, O8 + pd.Timedelta(weeks=13), color="#E3E9F3", lw=0)
        ax.plot(win.index, win.values, color=INK, lw=2.2, label="Observed")
        ax.plot(s.index, s.sn, color=GREY, lw=1.6, ls=":", label="Same week last year")
        ax.plot(s.index, s.rep, color=RED, lw=2.6, label="Our model")
        srch = r.role.startswith("search")
        if srch: ax.plot(s.index, s.rep_s, color=NAVY, lw=2.6, ls="--", label="Our model + search")
        ax.axvline(O8, color=INK, lw=1, ls=":")
        ax.set_title(f"{TT[r.role]}:  {EN[u]}", loc="left", color=NAVY, fontweight="bold", fontsize=21)
        txt = (f"error {r.wape_rep:.1f}% → {r.wape_rep_search:.1f}% with search" if srch else f"error {r.wape_rep:.1f}%  (last year {r.wape_lastyear:.1f}%)")
        ax.text(0.015, 0.04, txt, transform=ax.transAxes, fontsize=18, color=NAVY if srch else RED, fontweight="bold")
        lo, hi = min(win.min(), s[["sn", "rep", "rep_s"]].min().min()), max(win.max(), s[["sn", "rep", "rep_s"]].max().max())
        ax.set_ylim(lo - (hi - lo) * .28, hi + (hi - lo) * .12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 1, 3, 5, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=16); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    h1, l1 = axs[1, 0].get_legend_handles_labels()
    f.legend(h1, l1, loc="upper center", ncol=4, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 1.035))
    f.text(0.005, 0.5, "Daily arrivals, age 20–39", rotation=90, va="center", fontsize=19)
    f.tight_layout(rect=(0.015, 0, 1, 0.98)); save(f, "S8_cases.png")
except FileNotFoundError as e:
    print("S8 생략:", e)
print("저장:", sorted(p.name for p in OUT.glob("*.png")))
